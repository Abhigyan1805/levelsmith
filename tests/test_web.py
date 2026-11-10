"""The page's JavaScript rules must agree with the Python rules."""

import json
import os
import random
import re
import shutil
import subprocess

import pytest

from levelsmith.baseline import random_level_text
from levelsmith.game import DIRS, make_mover, parse_level

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "web", "index.template.html")

RUNNER = """
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = cases.map(({rows, moves}) => {
  const lv = parseLevel(rows);
  let s = initialState(lv);
  const trace = [];
  for (const d of moves) {
    const res = slide(lv, s, d);
    if (!res) { trace.push(null); continue; }
    s = res.state;
    trace.push([s.pos, [...s.got].sort((a, b) => a - b), [...s.baked].sort((a, b) => a - b), isGoal(lv, s)]);
  }
  return trace;
});
console.log(JSON.stringify(out));
"""


def core_js() -> str:
    with open(TEMPLATE) as f:
        html = f.read()
    m = re.search(r"// <core>(.*?)// </core>", html, re.S)
    assert m, "core block markers missing from template"
    return m.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_rules_match_python_rules():
    rng = random.Random(42)
    cases = []
    for _ in range(150):
        w, h = rng.randint(4, 9), rng.randint(4, 9)
        rows = random_level_text(rng, w, h).split("\n")
        moves = "".join(rng.choice("UDLR") for _ in range(30))
        cases.append({"rows": rows, "moves": moves})

    expected = []
    for case in cases:
        level = parse_level("\n".join(case["rows"]))
        move = make_mover(level)
        s = level.initial_state()
        trace = []
        for d in case["moves"]:
            t = move(s, d)
            if t is None:
                trace.append(None)
                continue
            s = t
            got = [c for i, c in enumerate(level.coals) if s[1] >> i & 1]
            baked = [c for i, c in enumerate(level.clay) if not s[2] >> i & 1]
            trace.append([s[0], got, baked, level.is_goal(s)])
        expected.append(trace)

    proc = subprocess.run(
        ["node", "-e", core_js() + RUNNER],
        input=json.dumps(cases), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == expected


def test_template_is_self_contained():
    with open(TEMPLATE) as f:
        html = f.read()
    assert "/*LEVELS*/[]" in html and "/*SOURCES*/[]" in html
    # No network resources: the page must work when opened from disk.
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    assert "@import" not in html and "fetch(" not in html
    assert set(DIRS) == set("UDLR")


# A very small DOM stand-in: enough to run the page script end to end in node
# (there is no browser in CI), load a level, and play it with key presses.
DOM_STUB = r"""
function el(tag) {
  const e = {
    tagName: (tag || 'div').toUpperCase(), children: [], style: {}, dataset: {},
    innerHTML: '', className: '', disabled: false, listeners: {},
    classList: {
      toggle(c, on) { const s = new Set(e.className.split(' ').filter(Boolean)); on ? s.add(c) : s.delete(c); e.className = [...s].join(' '); },
      remove(c) { this.toggle(c, false); }, add(c) { this.toggle(c, true); },
    },
    appendChild(ch) { e.children.push(ch); if (e.value === undefined && ch.value !== undefined) e.value = ch.value; return ch; },
    addEventListener(t, f) { e.listeners[t] = f; },
    get options() { return e.children; },
  };
  let text = '';
  Object.defineProperty(e, 'textContent', {
    get: () => text,
    set: (v) => { text = v; e.children.length = 0; },  // like the DOM: replaces children
  });
  return e;
}
const byId = {};
const saved = {};
globalThis.document = {
  getElementById(id) { if (!byId[id]) { byId[id] = el('div'); if (id === 'band') byId[id].value = 'all'; } return byId[id]; },
  createElement: el,
  querySelectorAll() { return []; },
};
const winListeners = {};
globalThis.window = {
  addEventListener(t, f) { winListeners[t] = f; },
  matchMedia() { return { matches: true }; },
};
globalThis.localStorage = { getItem: (k) => saved[k] || null, setItem: (k, v) => { saved[k] = v; } };
const key = (k) => winListeners.keydown({ key: k, target: { tagName: 'BODY' }, preventDefault() {} });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
"""

DOM_PLAY = r"""
(async () => {
  const out = [];
  const arrows = { U: 'ArrowUp', D: 'ArrowDown', L: 'ArrowLeft', R: 'ArrowRight' };
  for (let i = 0; i < visible.length; i++) {
    load(i);
    const m = visible[i];
    for (const d of m.sol) { key(arrows[d]); await sleep(30); }
    await sleep(30);
    out.push({ id: m.id, moves: byId.moves.textContent, msg: byId.message.textContent,
               cls: byId.message.className, listed: byId['level-list'].children.length });
  }
  // Undo and restart on the first level.
  load(0);
  key(arrows[visible[0].sol[0]]); await sleep(40);
  const afterMove = byId.moves.textContent;
  key('z'); await sleep(10);
  const afterUndo = byId.moves.textContent;
  key(arrows[visible[0].sol[0]]); await sleep(40); key('r'); await sleep(10);
  console.log(JSON.stringify({ out, afterMove, afterUndo, afterRestart: byId.moves.textContent,
                               progress: Object.keys(JSON.parse(saved['kiln-progress'] || '{}')).length }));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_script_plays_levels_to_a_win():
    from levelsmith.baseline import generate

    levels = []
    for i, band in enumerate(["easy", "medium", "hard"]):
        r = generate(band, 7, 6, seed=100 + i, seen=set())
        levels.append({"id": f"{band}-00", "src": "random", "band": band, "w": 7, "h": 6,
                       "rows": r["level"].split("\n"), "opt": r["features"]["optimal_moves"],
                       "sol": r["features"]["solution"]})
    sources = [{"id": "random", "label": "Random", "note": "note."}]
    with open(TEMPLATE) as f:
        html = f.read()
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    script = script.replace("/*LEVELS*/[]", json.dumps(levels))
    script = script.replace("/*SOURCES*/[]", json.dumps(sources))
    proc = subprocess.run(
        ["node", "-e", DOM_STUB + script + DOM_PLAY],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout)
    assert len(res["out"]) == 3
    for lvl, o in zip(levels, res["out"]):
        assert o["moves"] == lvl["opt"], o
        assert o["cls"] == "win" and "optimum" in o["msg"], o
        assert o["listed"] == 3
    assert (res["afterMove"], res["afterUndo"], res["afterRestart"]) == (1, 0, 0)
    assert res["progress"] == 3
