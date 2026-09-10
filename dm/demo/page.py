"""The one page both surfaces render, live and offline.

There is a single HTML template, a single event applier and a single renderer.
The live server feeds it events over SSE as the wire delivers them; the static
gallery feeds it the same events, replayed from a captured record, and says on
its face that it is a replay. Two templates would be two places for the badge
logic to drift, and the badge is the claim.

**Where the badge comes from.** `record.source == "rp2040"` *and*
`device.matches_reference` -- both, computed in Python by `DemoRecord.on_silicon`
and passed to the page as one boolean. The page never recomputes it and there is
no code path that sets it from anything else, which is what makes
`tests/test_demo.py` able to assert that a native rehearsal cannot show it.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The aggregate silicon timing, quoted in the methodology panel and nowhere
#: near a single drawing. `docs/claim4-bringup.md` §11.1: normal one-HALT output
#: carries no per-program cycle count, so labelling any of these as the selected
#: drawing's measured latency would be inventing a measurement.
CYCLE_TABLE = [
    {"corpus": "QuickDraw", "bytes": 157.0, "cycles": 7333.9, "ms": 0.611, "cpi": 1.959},
    {"corpus": "Tier A L1", "bytes": 100.8, "cycles": 7063.5, "ms": 0.589, "cpi": 1.921},
    {"corpus": "Tier A L0", "bytes": 43.5, "cycles": 2633.7, "ms": 0.219, "cpi": 1.903},
]

FOOTPRINT = {"flash": 1862, "static_ram": 0, "peak_stack": 492,
             "traces": 12670, "clock_mhz": 12}


def render(mode: str, *, checkpoint: dict, records: list[dict],
           title: str = "drawing-machine") -> str:
    """The complete self-contained page.

    `mode` is `live` or `gallery`. In `live` the ask panel posts to the server
    and events arrive over SSE; in `gallery` the same panels replay embedded
    records and the ask panel is replaced by the record list.
    """
    payload = {
        "mode": mode,
        "checkpoint": checkpoint,
        "records": records,
        "cycles": CYCLE_TABLE,
        "footprint": FOOTPRINT,
    }
    return _TEMPLATE.replace("__TITLE__", title).replace(
        "__PAYLOAD__", json.dumps(payload)
    )


def write(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
/* Paper, deliberately.

   This page is recorded once, as `docs/media/hero.gif`, and that recording sits
   beside four figures that `dm/demo/figures.py` draws from the same records:
   white ground, hairline ink, a serif caption under each. So the page is set
   the same way -- one column, a serif face, rules instead of cards -- and it is
   light only. A page that took its colours from the recorder's OS theme would
   produce a hero that does not match the figures it is printed next to, and the
   theme of the machine that happened to be on the bench is not a design.

   Two colours carry meaning and nothing else does: green is a verified silicon
   frame, red is ink the human supplied rather than the model. Red means the
   same thing here as it does in Figure 5. */
:root{
  --paper:#fdfdfb; --ink:#18181b; --muted:#79797f; --rule:#dcdad3;
  --ok:#2c6a45; --given:#c43a2e; --warn:#9a3b1e; --hot:#efece2;
  --serif:"Times New Roman",Times,Georgia,"Iowan Old Style",serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0 auto;max-width:900px;padding:0 30px 56px;background:var(--paper);
  color:var(--ink);font-family:var(--serif);font-size:16.5px;line-height:1.55;
  -webkit-font-smoothing:antialiased}
a{color:inherit}

header{padding:54px 0 26px;text-align:center;border-bottom:1px solid var(--rule)}
h1{margin:0;font-size:27px;font-weight:400;letter-spacing:-.005em}
.sub{margin:10px auto 0;max-width:62ch;color:var(--muted);font-size:15px}
.chips{margin-top:16px;color:var(--muted);font-family:var(--mono);font-size:11.5px;
  line-height:2}
.chip+.chip::before{content:"·";padding:0 7px;color:var(--rule)}

main{display:block}
.panel{padding:30px 0 24px;border-bottom:1px solid var(--rule)}
.panel h2{margin:0 0 18px;font-family:var(--mono);font-size:11px;font-weight:400;
  letter-spacing:.16em;text-transform:uppercase;color:var(--muted);text-align:center}

input[type=text]{display:block;width:100%;max-width:430px;margin:0 auto;padding:8px 10px;
  font-family:var(--serif);font-size:18px;text-align:center;color:var(--ink);
  background:transparent;border:0;border-bottom:1px solid var(--ink);border-radius:0}
input[type=text]:focus{outline:none;border-bottom-width:2px;margin-bottom:-1px}
button{font-family:var(--serif);font-size:14px;padding:5px 12px;border-radius:2px;
  border:1px solid var(--rule);background:transparent;color:var(--ink);cursor:pointer}
button:hover:not(:disabled){border-color:var(--ink)}
button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.row{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:14px}

/* Every explanatory line on this page is a figure caption, so all of them are
   set as one: small, centred, measured, under the thing they describe. */
.note{margin:16px auto 0;max-width:76ch;font-size:13.5px;color:var(--muted);
  text-align:center}
.note strong{color:var(--ink);font-weight:600}
.refusal{margin:14px auto 0;max-width:70ch;padding-top:12px;
  border-top:1px solid var(--warn);color:var(--warn);font-size:14px;text-align:center}

#canvasWrap{position:relative;aspect-ratio:1;width:100%;max-width:470px;margin:0 auto}
svg#canvas{width:100%;height:100%;display:block}
.stage{display:flex;justify-content:center;gap:20px;flex-wrap:wrap;margin-top:14px;
  font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.stage.spread{justify-content:space-between;gap:10px}
.badge{font-family:var(--mono);font-size:11.5px;letter-spacing:.03em;color:var(--muted)}
.badge.ok{color:var(--ok)}
.badge.ok::before{content:"●  ";font-size:9px;vertical-align:2px}
.badge.bad{color:var(--warn)}
.badge.bad::before{content:"○  ";font-size:9px;vertical-align:2px}

pre{margin:0;font-family:var(--mono);font-size:11.5px;line-height:1.6;overflow-x:auto}
#uart{max-height:200px;overflow-y:auto;white-space:pre;color:var(--muted);font-size:11px}
#hex{max-width:760px;margin:0 auto;max-height:130px;overflow-y:auto;word-break:break-all;
  white-space:pre-wrap;color:var(--muted);line-height:1.8}
#hex b{color:var(--ink);font-weight:400;background:var(--hot)}
ol#disasm{max-width:760px;margin:20px auto 0;padding:0 0 0 46px;font-family:var(--mono);
  font-size:11.5px;max-height:280px;overflow-y:auto;line-height:1.75;color:var(--ink)}
ol#disasm li{padding:0 5px;white-space:pre}
ol#disasm li.hot{background:var(--hot);font-weight:600}
ol#disasm li.done{color:var(--muted)}

table{border-collapse:collapse;width:100%;max-width:760px;margin:0 auto;font-size:13px}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
td{font-family:var(--mono);font-size:11.5px}
th{color:var(--muted);font-weight:400;font-size:11px;letter-spacing:.1em;
  text-transform:uppercase}

/* Figure 4 on paper, live: the device drawing in ink, the training drawings it
   is nearest in grey, because one of them came off a wire and three came out of
   a corpus and the sheet must not let a reader mix them up. */
.nnGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:18px;
  max-width:760px;margin:0 auto}
.nnCell{text-align:center}
.nnCell svg{width:100%;aspect-ratio:1;color:var(--ink)}
.nnCell:not(:first-child) svg{color:#9d9da5}
.nnCell .cap{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:7px}
.floorBar{position:relative;height:1px;max-width:760px;margin:26px auto 0;
  background:var(--rule)}
.floorBar .fill{display:none}
.floorBar .mark{position:absolute;top:-6px;height:13px;width:1.5px;background:var(--ink)}

.recList{display:flex;flex-wrap:wrap;justify-content:center;gap:5px;max-width:820px;
  margin:0 auto}
.recList button{font-family:var(--mono);font-size:11px;padding:4px 8px;color:var(--muted)}
.recList button.sel{border-color:var(--ink);color:var(--ink)}
.pipe{font-family:var(--mono);font-size:11.5px;color:var(--muted);line-height:1.95;
  white-space:pre-wrap;word-break:break-word;text-align:center}
.pipe b{color:var(--ink);font-weight:400}
footer{padding:26px 0 0;color:var(--muted);font-size:12.5px;text-align:center}
</style>
</head>
<body>
<header>
  <h1>drawing-machine — <span id="hTitle">a word, a program, a microcontroller</span></h1>
  <p class="sub" id="hSub"></p>
  <div class="chips" id="hChips"></div>
</header>

<main>
  <section class="panel" id="askPanel">
    <h2 id="askHead">Ask</h2>
    <div id="askBody"></div>
    <div id="routing" class="note"></div>
    <div id="refusal"></div>
  </section>

  <section class="panel">
    <h2>The drawing — geometry from the device trace</h2>
    <div id="canvasWrap">
      <svg id="canvas" viewBox="0 0 256 256" preserveAspectRatio="xMidYMid meet"
           aria-label="the drawing the device produced">
        <g id="ink" fill="none" stroke="currentColor" stroke-linecap="round"
           stroke-linejoin="round"></g>
      </svg>
    </div>
    <div class="stage">
      <span id="stage">idle</span>
      <span id="badge" class="badge">no run yet</span>
    </div>
    <div class="note" id="canvasNote"></div>
  </section>

  <section class="panel">
    <h2>The program the model emitted</h2>
    <pre id="hex">—</pre>
    <ol id="disasm"></ol>
    <div class="note" id="progNote"></div>
  </section>

  <section class="panel">
    <h2>The wire — what came back from the board</h2>
    <pre id="uart">—</pre>
    <div class="stage"><span id="devStats"></span></div>
    <div class="note" id="devNote"></div>
  </section>

  <section class="panel" id="novPanel">
    <h2>Is it new?</h2>
    <div id="novBody" class="note">not measured for this record</div>
  </section>

  <section class="panel">
    <h2>How to read this page</h2>
    <div class="pipe" id="pipe"></div>
    <div style="margin-top:20px" id="cycleTable"></div>
  </section>
</main>

<footer id="foot"></footer>

<script>
const DATA = __PAYLOAD__;
const CANVAS = 256;
const $ = id => document.getElementById(id);

let view = null;      // the record currently on screen
let timer = null;     // replay timer
let source = null;    // live EventSource

/* ---------- header ------------------------------------------------------ */
function header(){
  const c = DATA.checkpoint;
  $("hSub").textContent = DATA.mode === "live"
    ? "Type a word. The model writes bytecode, the bytecode is flashed to an RP2040, and the drawing on this page is what the chip sent back."
    : "Captured runs. Every drawing below is geometry the RP2040 sent over a UART; nothing here is re-rendered from the host.";
  const f = DATA.footprint;
  const chips = [
    c.params.toLocaleString() + " parameters",
    c.codec + " codec · " + c.shape,
    c.classes.length + " categories: " + c.classes.join(", "),
    f.flash.toLocaleString() + " B flash · " + f.static_ram + " B static RAM · " + f.peak_stack + " B peak stack",
    f.traces.toLocaleString() + "/" + f.traces.toLocaleString() + " traces bit-exact on silicon",
  ];
  $("hChips").innerHTML = chips.map(t => '<span class="chip">' + esc(t) + '</span>').join("");
}

function esc(s){ return String(s).replace(/[&<>"]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m])); }

/* ---------- canvas ------------------------------------------------------ */
let ink, open_ = null, openPts = [], pointIndex = 0, strokeBase = 0;
/* Instructions below this index came from the held-out human prefix rather
   than from the model. -1 means the whole program is the model's. */
let prefixCut = -1;

function clearCanvas(){
  queue = [];
  if(drain){ clearInterval(drain); drain = null; }
  ink = $("ink");
  ink.innerHTML = "";
  open_ = null; openPts = []; pointIndex = 0; strokeBase = 0;
  $("uart").textContent = "";
  hot(-1);
}

function addPoint(x, y){
  if(!open_){
    open_ = document.createElementNS("http://www.w3.org/2000/svg","polyline");
    open_.setAttribute("stroke-width","1.6");
    open_.setAttribute("opacity","0.55");
    ink.appendChild(open_);
    openPts = [];
  }
  openPts.push(x.toFixed(2) + "," + y.toFixed(2));
  open_.setAttribute("points", openPts.join(" "));
  hot(pointIndex);
  pointIndex++;
}

function endStroke(points, width){
  if(open_) ink.removeChild(open_);
  open_ = null; openPts = [];
  const base = strokeBase;
  strokeBase += points.length;
  // With a held-out prefix the stroke is drawn in two colours: the part the
  // human drew and the part the model continued. The split is read from the
  // same verified alignment the highlight uses, so a stroke can only be
  // divided where the instruction stream actually divides it.
  for(const run of runs(points, base)){
    const el = document.createElementNS("http://www.w3.org/2000/svg","polyline");
    el.setAttribute("points", run.points.map(p => p[0].toFixed(2)+","+p[1].toFixed(2)).join(" "));
    el.setAttribute("stroke-width", String(Math.max(1, width)));
    // Red is the human's prefix, on this page and on the prefix figure both.
    if(run.human){ el.style.stroke = "var(--given)"; }
    ink.appendChild(el);
  }
}

function runs(points, base){
  if(prefixCut < 0 || !alignment) return [{points: points, human: false}];
  const map = alignment.instruction_of_point;
  const out = [];
  let current = null;
  for(let i = 0; i < points.length; i++){
    const instr = map[base + i];
    const human = instr !== undefined && instr < prefixCut;
    if(!current || current.human !== human){
      // The boundary point belongs to both runs, so the two meet rather than
      // leaving a gap. Only that point: sharing the whole segment would draw it
      // twice, and the second colour would claim ink the first one owns.
      if(current) current.points.push(points[i]);
      current = {points: [], human: human};
      out.push(current);
    }
    current.points.push(points[i]);
  }
  return out;
}

function addRegion(points){
  if(open_) ink.removeChild(open_);
  open_ = null; openPts = [];
  const el = document.createElementNS("http://www.w3.org/2000/svg","polygon");
  el.setAttribute("points", points.map(p => p[0].toFixed(2)+","+p[1].toFixed(2)).join(" "));
  el.setAttribute("fill","currentColor"); el.setAttribute("stroke","none");
  ink.appendChild(el);
}

function addDisc(d){
  const el = document.createElementNS("http://www.w3.org/2000/svg","circle");
  el.setAttribute("cx", d.cx); el.setAttribute("cy", d.cy); el.setAttribute("r", d.r);
  el.setAttribute("stroke-width", String(Math.max(1, d.width)));
  ink.appendChild(el);
}

/* ---------- program listing -------------------------------------------- */
let alignment = null, instrOffsets = [], programHex = "";

function listProgram(rec){
  const lines = (rec.sampled.disassembly || "").split("\n").filter(s => s.length);
  $("disasm").innerHTML = lines.map(l => "<li>" + esc(l) + "</li>").join("");
  programHex = rec.sampled.bytecode_hex || "";
  paintHex(-1);
  alignment = rec.device.alignment || null;
  instrOffsets = alignment ? alignment.offsets : [];
  const prefix = rec.prefix;
  $("progNote").innerHTML =
    (prefix ? "<strong>The first " + prefix.instructions + " instructions (" + prefix.bytes +
      " bytes) are a drawing from the validation split the model has never seen</strong>, " +
      "cut at an instruction boundary and handed over for it to finish. On the canvas the " +
      "human's part is drawn in red and the black ink is the model's continuation. " : "") +
    (alignment
      ? "The highlighted line is the instruction that emitted the point currently arriving. <strong>That highlight is computed on the host</strong> and checked against the device's own stroke lengths; the geometry is the device's."
      : "Instruction highlighting is off for this program: the host walker and the device disagreed, or the program uses a tier the walker refuses. The drawing is unaffected.");
}

function paintHex(upto){
  if(!programHex){ $("hex").textContent = "—"; return; }
  const cut = upto < 0 ? 0 : upto;
  const head = programHex.slice(0, cut*2), tail = programHex.slice(cut*2);
  $("hex").innerHTML = "<b>" + esc(head.replace(/(..)/g,"$1 ")) + "</b>" + esc(tail.replace(/(..)/g,"$1 "));
}

function hot(pointIdx){
  const items = $("disasm").children;
  for(const li of items) li.className = "";
  if(!alignment || pointIdx < 0) { paintHex(-1); return; }
  const idx = alignment.instruction_of_point[pointIdx];
  if(idx === undefined) return;
  for(let i=0;i<items.length;i++) if(i < idx) items[i].className = "done";
  if(items[idx]){ items[idx].className = "hot"; items[idx].scrollIntoView({block:"nearest"}); }
  paintHex(instrOffsets[idx] !== undefined ? instrOffsets[idx] : -1);
}

/* ---------- pacing -------------------------------------------------------
   The wire is faster than an eye. A QuickDraw program's whole trace is about a
   kilobyte, which crosses a 115200-baud line in under a tenth of a second, and
   the drawing would simply appear. So the live view can queue the events and
   drain them at a watchable rate.

   This changes only *when* a point is drawn, never which point: every
   coordinate is the device's, in the order the device sent it. The page states
   the pacing and states the real elapsed time beside it, because a demo that
   slowed a device down without saying so would be misrepresenting the one
   number this whole project is about. */
let paced = true, queue = [], drain = null;

function enqueue(e){
  if(!paced || e.t === "record" || e.t === "error" || e.t === "saved"){
    applyEvent(e);
    return;
  }
  queue.push(e);
  if(!drain){
    drain = setInterval(() => {
      if(!queue.length){ clearInterval(drain); drain = null; return; }
      applyEvent(queue.shift());
    }, 16);
  }
}

/* ---------- events ------------------------------------------------------ */
function applyEvent(e){
  if(e.t === "loading"){ $("stage").textContent = "writing " + e.bytes + " bytes of bytecode to the board — wait for the remount, do not unplug"; }
  else if(e.t === "loaded"){ $("stage").textContent = "image loaded in " + e.seconds + " s — waiting for the frame"; }
  else if(e.t === "begin"){ $("stage").textContent = "frame open — frac_bits " + e.frac_bits + ", curve_steps " + e.curve_steps; }
  else if(e.t === "point"){ addPoint(e.x, e.y); }
  else if(e.t === "stroke"){ endStroke(e.points, e.width); }
  else if(e.t === "region"){ addRegion(e.points); }
  else if(e.t === "disc"){ addDisc(e); }
  else if(e.t === "fault"){ $("stage").textContent = "device reported fault " + e.kind + " at pc " + e.pc; }
  else if(e.t === "line"){ appendUart(e.text); }
  else if(e.t === "done"){ finish(e); }
  else if(e.t === "record"){ showRecord(e.record); }
  else if(e.t === "error"){ fail(e.message); }
}

function appendUart(text){
  const el = $("uart");
  el.textContent += (el.textContent ? "\n" : "") + text;
  el.scrollTop = el.scrollHeight;
}

function finish(e){
  hot(-1);
  $("stage").textContent = "frame complete"
    + (e.seconds ? " — " + e.seconds + " s from UF2 write to last byte on the wire" : "");
  $("devStats").textContent =
    "steps " + e.steps + " · halted " + (e.halted ? "yes" : "no") +
    " · strokes " + e.strokes + " · faults " + (e.faults ? e.faults.length : 0);
}

function fail(message){
  // The canvas is emptied, not dimmed. A page that kept unverified geometry on
  // screen beside the word "refused" would still be showing a drawing it
  // cannot vouch for, and a viewer reads the picture before the caption.
  if(queue) queue.length = 0;
  if(drain){ clearInterval(drain); drain = null; }
  if(ink) ink.innerHTML = "";
  open_ = null; openPts = []; pointIndex = 0; strokeBase = 0;
  $("badge").className = "badge bad";
  $("badge").textContent = "refused";
  $("stage").textContent = "refused — nothing drawn";
  $("canvasNote").innerHTML =
    "<strong>Nothing is drawn here on purpose.</strong> The frame did not pass " +
    "its check, so this page has no verified geometry to show. It does not fall " +
    "back to the host reference VM's drawing, which is the whole point of the " +
    "check.";
  $("refusal").innerHTML = '<div class="refusal">' + esc(message) + "</div>";
}

/* ---------- one record on screen ---------------------------------------- */
function showRecord(rec, animate){
  view = rec;
  clearCanvas();
  $("refusal").innerHTML = "";
  listProgram(rec);
  routingNote(rec);
  novelty(rec);
  prefixCut = rec.prefix ? rec.prefix.instructions : -1;
  const silicon = rec.on_silicon;
  const b = $("badge");
  b.className = "badge " + (silicon ? "ok" : "bad");
  b.textContent = silicon
    ? "RP2040 trace: exact match"
    : (rec.source === "rp2040" ? "RP2040 trace: DIVERGED — refused" : "host rehearsal — not silicon");
  const replayed = DATA.mode === "gallery"
    ? "<strong>A replay of a captured trace, not a live run.</strong> The points are the ones that were sent; the pace is chosen so a human can watch them. "
    : "";
  $("canvasNote").innerHTML = silicon
    ? replayed + "Every coordinate above arrived over a 115200-baud UART from a Raspberry Pi Pico and equals the reference VM's trace exactly. The host reference computed the equality check and drew nothing."
    : replayed + (rec.source === "rp2040"
        ? "This frame did not equal the reference. It is shown because a refusal is evidence; it is not a result."
        : "<strong>No board was involved in this record.</strong> It was produced by the native harness so the page could be built and rehearsed without hardware, and it carries no silicon claim.");
  $("devNote").innerHTML = "Raw lines as received. <span style='font-family:var(--mono)'>p</span> is a point in units of 1/4096 canvas pixel, <span style='font-family:var(--mono)'>e stroke</span> closes a path, <span style='font-family:var(--mono)'>=</span> is the per-program verdict.";
  if(animate){ replay(rec); }
  else { staticDraw(rec); }
}

function staticDraw(rec){
  const g = rec.device.geometry || {strokes:[],regions:[],discs:[]};
  g.regions.forEach(r => addRegion(r.points));
  g.strokes.forEach(s => endStroke(s.points, s.width));
  (g.discs||[]).forEach(addDisc);
  $("uart").textContent = rec.device.trace_text || "—";
  finish(rec.device);
}

function replay(rec){
  if(timer) { clearInterval(timer); timer = null; }
  const g = rec.device.geometry || {strokes:[],regions:[],discs:[]};
  const steps = [];
  const lines = (rec.device.trace_text || "").split("\n");
  let li = 0;
  // Interleave the raw lines with the geometry they encode, so the log and the
  // canvas advance together rather than the log arriving all at once.
  g.strokes.forEach(s => {
    s.points.forEach(p => steps.push({t:"point", x:p[0], y:p[1]}));
    steps.push({t:"stroke", points:s.points, width:s.width});
  });
  g.regions.forEach(r => steps.push({t:"region", points:r.points}));
  (g.discs||[]).forEach(d => steps.push(Object.assign({t:"disc"}, d)));
  steps.push({t:"done", steps:rec.device.steps, halted:rec.device.halted,
              strokes:(g.strokes||[]).length, faults:rec.device.faults||[]});
  let i = 0;
  $("stage").textContent = "replaying " + steps.length + " captured events";
  const perLine = Math.max(1, Math.ceil(lines.length / Math.max(1, steps.length)));
  timer = setInterval(() => {
    if(i >= steps.length){ clearInterval(timer); timer = null; return; }
    applyEvent(steps[i++]);
    for(let k=0;k<perLine && li<lines.length;k++) appendUart(lines[li++]);
  }, 18);
}

/* ---------- routing + novelty ------------------------------------------- */
function routingNote(rec){
  const r = rec.routing;
  const sampled = rec.sampled;
  $("routing").innerHTML =
    "<strong>" + esc(r.reason) + "</strong><br>" +
    (rec.prefix ? "continuing a held-out drawing (" + rec.prefix.instructions + " of " +
      rec.prefix.of_instructions + " instructions given)<br>" : "") +
    "seed " + sampled.seed + " · top-k " + (sampled.top_k === null ? "full" : sampled.top_k) +
    " · T " + sampled.temperature + " · " + sampled.bytes + " bytes · " +
    sampled.instructions + " instructions · sampled in " + sampled.sample_seconds + " s";
}

function strokesSvg(strokes){
  if(!strokes || !strokes.length) return "";
  const body = strokes.map(s =>
    '<polyline points="' + s.points.map(p => p[0].toFixed(1)+","+p[1].toFixed(1)).join(" ") +
    '" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" ' +
    'stroke-linejoin="round"/>').join("");
  return '<svg viewBox="0 0 256 256">' + body + '</svg>';
}



function novelty(rec){
  const n = rec.novelty;
  if(!n || n.empty){ $("novBody").innerHTML = "Not measured for this record."; return; }
  const g = rec.device.geometry || {strokes:[]};
  const cells = [
    '<div class="nnCell">' + strokesSvg(g.strokes || []) +
    '<div class="cap">this drawing<br>from the device trace</div></div>'
  ].concat(n.neighbours.map(nb =>
    '<div class="nnCell">' + strokesSvg(nb.strokes) +
    '<div class="cap">nearest training #' + nb.index + '<br>' +
    nb.distance.toFixed(2) + ' px</div></div>'));
  const pct = n.floor_percentile;
  const width = Math.max(0, Math.min(100, pct));
  $("novBody").innerHTML =
    '<div class="nnGrid">' + cells.join("") + '</div>' +
    '<div style="margin-top:14px">' +
      '<strong>Verbatim copy of a training program: ' + (n.verbatim ? "YES" : "no") + '</strong> — ' +
      'checked against all ' + n.train_n.toLocaleString() + ' programs the model was trained on.' +
      (n.longest_shared_run === undefined ? "" :
        ' The longest run of bytes it shares with any training program of this category is <strong>' +
        n.longest_shared_run + ' of ' + rec.sampled.bytes + '</strong> — so it is not a new drawing ' +
        'assembled out of memorised fragments either.') +
    '</div>' +
    '<div style="margin-top:10px">Nearest training drawing of this category: <strong>' +
      n.distance.toFixed(2) + ' px</strong> (Chamfer, over ' + n.bank_n.toLocaleString() + ' drawings). ' +
      'Held-out human drawings of the same category sit a median <strong>' + n.floor_median.toFixed(2) +
      ' px</strong> from the same set — that is the scale this number is read on. ' +
      'This sample is at the <strong>' + pct.toFixed(0) + 'th percentile</strong> of that held-out distribution.' +
    '</div>' +
    '<div class="floorBar"><div class="fill" style="width:' + width + '%"></div>' +
      '<div class="mark" style="left:' + width + '%"></div>' +
      '<div class="mark" style="left:50%;background:var(--muted);opacity:.6"></div></div>' +
    '<div class="stage spread"><span>0th — closer to training data than any held-out human drawing</span>' +
      '<span>median</span><span>100th — farther than all of them</span></div>' +
    '<div class="note">The marker is this sample; the faint tick is the held-out median. ' +
    'It says the drawing is not a copy. It does not say it is a <em>good</em> drawing — ' +
    'that is a set-level question, measured separately in <span style="font-family:var(--mono)">docs/conditioning.md</span> §6, ' +
    'which found this sampler sits in each class\'s typical middle.</div>';
}

/* ---------- methodology -------------------------------------------------- */
function methodology(){
  $("pipe").innerHTML =
    "<b>word</b> → router → <b>825k-parameter transformer</b> → <b>bytecode</b> → UF2 → SRAM →\n" +
    "<b>RP2040 VM</b> → UART at 115200 → this page's parser → the picture above\n\n" +
    "The host reference VM computes the equality assertion and never supplies geometry. " +
    "A mismatch, a fault, an incomplete frame or a failed BOOTSEL return refuses the page " +
    "rather than falling back to a host drawing.";
  const f = DATA.footprint;
  const rows = DATA.cycles.map(r =>
    "<tr><td>" + r.corpus + "</td><td>" + r.bytes.toFixed(1) + "</td><td>" +
    r.cycles.toLocaleString() + "</td><td>" + r.ms.toFixed(3) + " ms</td><td>" +
    r.cpi.toFixed(3) + "</td></tr>").join("");
  $("cycleTable").innerHTML =
    "<table><thead><tr><th>corpus</th><th>bytes/drawing</th><th>cycles/drawing</th>" +
    "<th>at " + f.clock_mhz + " MHz</th><th>cycles/instruction</th></tr></thead><tbody>" +
    rows + "</tbody></table>" +
    "<div class='note'>Aggregate timing over 40 programs per corpus, minimum of five repetitions, " +
    "13 cycles of instrument overhead subtracted, executing from zero-wait-state SRAM. " +
    "<strong>It is not this drawing's measured latency</strong> — a normal one-HALT run emits no " +
    "per-program cycle count, so no number here is attached to the picture above.</div>";
  $("foot").innerHTML =
    "Model: 825,344 parameters, byte codec, trained on five QuickDraw categories. " +
    "Device: Raspberry Pi Pico (RP2040, Cortex-M0+) executing from SRAM, traced over an " +
    "ELEGOO UNO R3 held in reset as a USB-UART bridge. " +
    "Energy per drawing is explicitly not measured and not claimed.";
}

/* ---------- live mode ---------------------------------------------------- */
function liveUI(){
  $("askBody").innerHTML =
    '<input type="text" id="word" placeholder="a word — try cat, bus, flower, sailboat, bicycle" autocomplete="off">' +
    '<div class="row"><button class="primary" id="go">Draw it</button>' +
    '<button id="again">New sample</button>' +
    '<button id="pace">pace: watchable</button></div>' +
    '<div class="row" id="quick"></div>' +
    '<div class="note" id="paceNote">Points are drawn at a rate a person can ' +
    'follow. Every coordinate and its order are the device\'s; only the timing ' +
    'is the page\'s, and the real elapsed time is reported when the frame ' +
    'closes.</div>';
  $("quick").innerHTML = DATA.checkpoint.classes
    .map(c => '<button data-w="' + c + '">' + c + '</button>').join("");
  $("quick").addEventListener("click", ev => {
    const w = ev.target.getAttribute("data-w");
    if(w){ $("word").value = w; draw(w); }
  });
  $("pace").onclick = () => {
    paced = !paced;
    $("pace").textContent = "pace: " + (paced ? "watchable" : "as it arrives");
    $("paceNote").style.display = paced ? "" : "none";
  };
  $("go").onclick = () => draw($("word").value);
  $("again").onclick = () => draw($("word").value, true);
  $("word").addEventListener("keydown", e => { if(e.key === "Enter") draw($("word").value); });
}

let busy = false;
function draw(word, resample){
  if(busy || !word) return;
  busy = true;
  $("go").disabled = true; $("again").disabled = true;
  clearCanvas();
  $("refusal").innerHTML = "";
  $("badge").className = "badge"; $("badge").textContent = "running…";
  $("stage").textContent = "sampling…";
  if(source) source.close();
  const url = "/api/draw?word=" + encodeURIComponent(word) + (resample ? "&resample=1" : "");
  source = new EventSource(url);
  source.onmessage = ev => {
    const e = JSON.parse(ev.data);
    enqueue(e);
    if(e.t === "record" || e.t === "error"){
      source.close(); source = null; busy = false;
      $("go").disabled = false; $("again").disabled = false;
    }
  };
  source.onerror = () => {
    if(source){ source.close(); source = null; }
    busy = false; $("go").disabled = false; $("again").disabled = false;
    fail("the demo server stopped sending. Is the board still attached?");
  };
}

/* ---------- gallery mode -------------------------------------------------- */
function galleryUI(){
  $("askHead").textContent = "Captured runs";
  const list = document.createElement("div");
  list.className = "recList";
  list.innerHTML = DATA.records.map((r,i) => {
    const label = r.prefix
      ? esc(r.routing.class || "?") + " · continuation · " + r.sampled.bytes + " B"
      : esc(r.word) + (r.routing.rule === "synonym" ? " → " + esc(r.routing.class) : "") +
        " · seed " + r.sampled.seed + " · " + r.sampled.bytes + " B";
    const mark = r.on_silicon ? "" : " ·  ⚠";
    return '<button data-i="' + i + '">' + label + mark + "</button>";
  }).join("");
  $("askBody").appendChild(list);
  list.addEventListener("click", ev => {
    const i = ev.target.getAttribute("data-i");
    if(i === null) return;
    for(const b of list.children) b.className = "";
    ev.target.className = "sel";
    showRecord(DATA.records[+i], true);
  });
  if(DATA.records.length){
    list.children[0].className = "sel";
    showRecord(DATA.records[0], true);
  }

}

/* ---------- boot ---------------------------------------------------------- */
header();
methodology();
clearCanvas();
if(DATA.mode === "live"){ liveUI(); } else { galleryUI(); }
</script>
</body>
</html>
"""
