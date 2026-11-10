"""Minimal LLM clients (stdlib only) with budget caps and per-call logging.

Two providers: Gemini (generateContent REST API) and any OpenAI-compatible
chat endpoint (used for DeepSeek). Keys are read from the environment or from
a KEY=VALUE file and are never logged.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict

DEFAULT_ENV_FILE = os.path.expanduser("~/.config/fm-secrets/llm.env")

# USD per million tokens. Output includes reasoning tokens. Read from the
# providers' pricing pages on 2026-11-10 (see README). DeepSeek charges double
# during peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays). Gemini
# calls in this project ran on the free tier, so the Gemini figure is a
# list-price equivalent, not money spent.
PRICES = {
    "deepseek-flash": {"in": 0.15, "in_cached": 0.003, "out": 0.60, "peak_x": 2.0},
    "gemini-3.6-flash": {"in": 0.75, "in_cached": 0.075, "out": 3.75},
}
PRICES["gemini-3.5-flash"] = PRICES["gemini-3.6-flash"]  # served by 3.6-flash


def _deepseek_peak(now: float | None = None) -> bool:
    t = time.gmtime(now)
    return t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


def load_env(path: str = DEFAULT_ENV_FILE) -> dict[str, str]:
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.replace("export ", "").strip()] = v.strip().strip("\"'")
    for k in ("GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0  # visible + reasoning
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    http_status: int | None = None
    retries: int = 0
    error: str = ""


def _scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "<redacted>")
    return text


def cost_of(model: str, prompt: int, cached: int, output: int) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    fresh = max(prompt - cached, 0)
    usd = (fresh * p["in"] + cached * p["in_cached"] + output * p["out"]) / 1e6
    if "peak_x" in p and _deepseek_peak():
        usd *= p["peak_x"]
    return usd


class Budget:
    """Hard caps on number of calls and on spend, shared across a run.

    `reserve_usd` is the assumed worst-case cost of one call. It is held back
    while a call is in flight so that parallel workers cannot overshoot the cap.
    """

    def __init__(self, max_calls: int, max_usd: float, reserve_usd: float = 0.0):
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.reserve_usd = reserve_usd
        self.calls = 0
        self.usd = 0.0
        self._in_flight = 0
        self._lock = threading.Lock()

    def begin(self) -> None:
        with self._lock:
            if self.calls + self._in_flight >= self.max_calls:
                raise BudgetExceeded(f"call cap reached ({self.max_calls})")
            held = (self._in_flight + 1) * self.reserve_usd
            if self.usd + held > self.max_usd:
                raise BudgetExceeded(f"spend cap reached (${self.max_usd:.2f})")
            self._in_flight += 1

    def end(self, usd: float) -> None:
        with self._lock:
            self._in_flight -= 1
            self.calls += 1
            self.usd += usd


class LLMClient:
    """Base class: retry/backoff, budget, logging. Subclasses do the HTTP."""

    provider = "base"

    def __init__(
        self,
        model: str,
        budget: Budget,
        log_path: str | None = None,
        timeout_s: float = 150.0,
        max_retries: int = 4,
        min_interval_s: float = 0.0,
        max_output_tokens: int = 8000,
    ):
        self.model = model
        self.budget = budget
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.max_output_tokens = max_output_tokens
        self._next_slot = 0.0
        self._lock = threading.Lock()
        self._secrets: list[str] = []

    # -- provider hooks --
    def _request(self, system: str, messages: list[dict], temperature: float):
        raise NotImplementedError

    def _parse(self, payload: dict) -> LLMResponse:
        raise NotImplementedError

    def _pace(self) -> None:
        """Space out request starts by min_interval_s across all threads."""
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.time()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval_s
        if slot > now:
            time.sleep(slot - now)

    # -- public --
    def complete(
        self,
        system: str,
        messages: list[dict],
        temperature: float = 1.0,
        tag: dict | None = None,
    ) -> LLMResponse:
        """messages: [{"role": "user"|"assistant", "content": str}, ...]"""
        self.budget.begin()
        resp = LLMResponse(ok=False, provider=self.provider, model=self.model)
        t_start = time.perf_counter()
        attempt = 0
        try:
            url, headers, body = self._request(system, messages, temperature)
            data = json.dumps(body).encode()
            while True:
                self._pace()
                status, payload, err = None, None, ""
                try:
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json", **headers},
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                        status = r.status
                        payload = json.loads(r.read())
                except urllib.error.HTTPError as e:
                    status = e.code
                    err = e.read()[:2000].decode("utf8", "replace")
                except Exception as e:  # timeouts, connection errors, bad JSON
                    err = f"{type(e).__name__}: {e}"
                if payload is not None:
                    try:
                        resp = self._parse(payload)
                    except Exception as e:
                        resp.error = f"unparseable response: {type(e).__name__}: {e}"
                    resp.http_status = status
                    break
                resp.http_status = status
                resp.error = _scrub(err, self._secrets)[:500]
                retryable = status in (429, 500, 502, 503, 504) or status is None
                if not retryable or attempt >= self.max_retries:
                    break
                attempt += 1
                delay = min(60.0, 2.0 ** attempt)
                m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', err)
                if m:
                    delay = min(90.0, float(m.group(1)) + 1.0)
                time.sleep(delay)
            resp.retries = attempt
            resp.provider, resp.model = self.provider, self.model
            resp.latency_s = round(time.perf_counter() - t_start, 3)
            resp.cost_usd = cost_of(
                self.model, resp.prompt_tokens, resp.cached_tokens, resp.output_tokens
            )
        finally:
            self.budget.end(resp.cost_usd)
        self._log(resp, tag or {}, len(messages))
        return resp

    def _log(self, resp: LLMResponse, tag: dict, n_messages: int) -> None:
        if not self.log_path:
            return
        rec = {"ts": round(time.time(), 3), **tag, "n_messages": n_messages}
        rec.update(asdict(resp))
        del rec["text"]  # replies are stored with the results, not in the call log
        rec["budget_calls"] = self.budget.calls
        rec["budget_usd"] = round(self.budget.usd, 6)
        with self._lock:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        budget: Budget,
        thinking_budget: int | None = None,
        **kw,
    ):
        super().__init__(model, budget, **kw)
        self._key = api_key
        self._secrets = [api_key]
        self.thinking_budget = thinking_budget

    def _request(self, system, messages, temperature):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if self.thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget
            }
        return url, {"x-goog-api-key": self._key}, body

    def _parse(self, payload):
        u = payload.get("usageMetadata", {})
        cands = payload.get("candidates") or [{}]
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thoughts = u.get("thoughtsTokenCount", 0)
        return LLMResponse(
            ok=bool(text),
            text=text,
            prompt_tokens=u.get("promptTokenCount", 0),
            cached_tokens=u.get("cachedContentTokenCount", 0),
            output_tokens=u.get("candidatesTokenCount", 0) + thoughts,
            reasoning_tokens=thoughts,
            error="" if text else f"empty response: {cands[0].get('finishReason')}",
        )


class OpenAICompatClient(LLMClient):
    provider = "openai-compat"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        budget: Budget,
        provider: str = "deepseek",
        extra_body: dict | None = None,
        **kw,
    ):
        super().__init__(model, budget, **kw)
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._secrets = [api_key]
        self.provider = provider
        self.extra_body = extra_body or {}

    def _request(self, system, messages, temperature):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            **self.extra_body,
        }
        return (
            self._base + "/chat/completions",
            {"Authorization": "Bearer " + self._key},
            body,
        )

    def _parse(self, payload):
        u = payload.get("usage", {})
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
        cached = u.get("prompt_cache_hit_tokens") or (
            u.get("prompt_tokens_details") or {}
        ).get("cached_tokens", 0)
        return LLMResponse(
            ok=bool(text),
            text=text,
            prompt_tokens=u.get("prompt_tokens", 0),
            cached_tokens=cached or 0,
            output_tokens=u.get("completion_tokens", 0),
            reasoning_tokens=(u.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            ),
            error="" if text else f"empty response: {choice.get('finish_reason')}",
        )


def deepseek_balance(env: dict[str, str] | None = None) -> float | None:
    """Current DeepSeek account balance in USD, or None if unavailable."""
    env = env or load_env()
    try:
        req = urllib.request.Request(
            env["DEEPSEEK_BASE_URL"].rstrip("/") + "/user/balance",
            headers={"Authorization": "Bearer " + env["DEEPSEEK_API_KEY"]},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        for b in d.get("balance_infos", []):
            if b.get("currency") == "USD":
                return float(b["total_balance"])
    except Exception:
        return None
    return None


def make_client(name: str, budget: Budget, log_path: str | None, env=None, **kw) -> LLMClient:
    """name: 'gemini' or 'deepseek', optionally 'provider:model'."""
    env = env or load_env()
    provider, _, model = name.partition(":")
    if provider == "gemini":
        return GeminiClient(
            model or "gemini-3.6-flash", env["GOOGLE_API_KEY"], budget,
            log_path=log_path, **kw,
        )
    if provider == "deepseek":
        return OpenAICompatClient(
            model or "deepseek-flash", env["DEEPSEEK_API_KEY"],
            env["DEEPSEEK_BASE_URL"], budget, log_path=log_path, **kw,
        )
    raise ValueError(f"unknown provider {provider!r}")
