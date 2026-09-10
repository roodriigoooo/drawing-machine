# Per-field bit attribution — pricing an architectural prior before building it

> **Status: §§1–3 written 2026-08-13 before the readings in §4, and left as
> written. §3 says exactly which of its branches are predictions and which are
> confirmations of something already glimpsed while the instrument was being
> built, because two of them are.**
>
> `docs/direction.md` §7.2 asks for `dm/eval/redundancy.py` generalised from the
> planner's stroke decoder to every arm and codec, reporting bits by operand
> `Kind`. That is the deliverable. The *reason* it ranks where it does is one
> sentence of the same section: **what has produced results here is instruments
> that read a model against the ISA's own structure**, and every one of them was
> hours of work, needed no training, and changed a conclusion.

---

## 1. The question §7.2 is really asking

§7.2 arrives from a stacked-TCN sketch classifier and keeps one idea from it:
dilated causal convolutions buy hierarchical locality cheaply, so a
convolutional front end over the byte stream would make **"an instruction is 1–4
bytes" architecture rather than something the model infers from position.**

The section already prices that by analogy — giving the model the same
information *for free* is the typing axis, and `token` against `byte` reads
−0.67 on Tier A and +11.58 on Tier B multi, corpus-dependent and inside the
floor where the model has capacity to spare. So the pre-registered expectation
is "a conv front end buys about what typing buys, which is little".

**An analogy is not a bound, and a bound is available for one forward pass.**
At an opcode position only 14 of 256 byte values are legal. At an `XF` operand
only 8 are. Whatever probability mass a model puts on the other 242 is mass it
is spending *because it does not know the grid* — and it is exactly what an
architecture that supplied the grid could take back, no more and no less:

```
waste(position) = −log2 P(symbol ∈ legal set at this position)
```

That is achievable by renormalisation alone, with no retraining and no new
parameters — the same construction `dm/eval/redundancy.py` uses for its ISA
baseline, which is why the two agree on the one arm and alphabet where both are
defined. **It is the rule `docs/traps.md` already carries for opcodes — measure
the redundancy before building the thing that removes it — applied to an
architecture.**

---

## 2. What is built

`dm/eval/attribution.py`, `scripts/attribution.py`, `tests/test_attribution.py`.

**Three cuts, each partitioning every scored byte**, plus a fourth where a byte
is more than one symbol:

| cut | rows |
|---|---|
| `field` | `opcode`, `coord_x`, `coord_y`, `delta_x`, `delta_y`, `count`, `scalar`, `id`, `xf`, `unparsed` |
| `slot` | the operand's ordinal inside its own instruction |
| `opcode` | every byte charged to the instruction it belongs to |
| `bitplane` | `field[k]`, MSB first — `bit` alphabet only |

**Why `coord_x` and `coord_y` are separate rows.** The ISA pairs coordinates,
and y is predicted *after* x within the same instruction. A single `coord` row
averages that away, and it is the one number here that bears directly on
whether within-instruction locality is already being exploited.

**Why the `bitplane` cut exists.** Under the `bit` alphabet every symbol is
legal everywhere, so the waste measure is vacuous there — a bit model *cannot*
spend mass on an impossible symbol. And that is precisely the alphabet where the
byte grid is hardest to know. What can be seen instead is which bit of which
field costs: a model that has located the byte boundaries and roughly inferred a
coordinate is cheap in that coordinate's high bits and expensive in its low
ones; a model that has not found the grid is flat across the eight.

### 2.1 Two decisions that decide whether any of it means anything

**Every codec, and that is the generalisation that mattered.**
`dm/eval/redundancy.py` refuses everything but `byte` and is right to: its
feasible masks are over byte *values*, which only a stride-1 untyped alphabet
maps onto symbols one-to-one. Attribution has no such limit — a bytecode byte is
`codec.stride` symbols in any alphabet, and the legal-symbol set at a position
is a property of the codec that every codec can state. So the same decomposition
reads on all four, and the comparison between them is legal because **they
encode identical bytecode**: bits/drawing per field holds content fixed and
varies only the alphabet.

> **Two units, and only one crosses a codec.** Bits per *drawing* and per *byte*
> are comparable across alphabets. Bits per *symbol* is not — the `bit` codec
> has eight symbols per byte, so its per-symbol rate is a different denominator
> wearing the same name. `docs/traps.md` carries the rule; the module reports
> per-symbol inside an arm and never as a headline.

**One reduction, two arms.** The flat AR path and the planner's stroke-decoder
path produce rows and hand them to the same fold, for the reason
`dm/eval/redundancy.py` gives: two copies of a reduction, one per arm, is how a
comparison ends up measuring its own two implementations. The planner path
reuses `redundancy.planner_rows` rather than re-deriving the prompt layout —
one position out and every byte's bits are credited to its neighbour, which on
the `coord_x`/`coord_y` split is the difference between the finding and its
mirror image.

**And the reconciliation is what makes it believable.** All cuts partition every
scored byte, so all must sum to the same total and that total must be the arm's
own cost. The driver prints the spread; a decomposition that does not close is a
decomposition of something else.

---

## 3. What each outcome settles, written before §4

A branch per number, for `docs/conditioning.md` §6.3's reason.

**B1 — the waste column, on `byte`. A CONFIRMATION, not a prediction.** The
instrument was smoke-tested during construction on 100–200 QuickDraw val
programs of the `byte` and `bit` arms, and the waste column read 0.0000 there.
So this branch is not open and is not claimed as one; what §4 owes is the same
number at full scale, which is a check that the smoke reading was not an
artifact of a short split. **If it holds, the ceiling on §7.2's conv front end
is ~0 bits/drawing on this corpus and alphabet, and the proposal is answered
without being built.**

**B2 — the waste column, across alphabets. Open.** `byte` must spread mass over
256 values where 14 are legal; `token` is handed the opcode region by its
alphabet; `bit` cannot waste at all. If `byte` wastes materially more than
`token`, that difference *is* the mechanism of the typing axis, measured on a
column the aggregate could not see. If all four are ~0, the untyped alphabet has
fully learned the region and typing has nothing structural left to give.

**B3 — the waste column on a corpus with narrow fields. Open, and this is the
branch with the most room to fire.** QuickDraw is `MOVE`/`LINE`/`HALT` and its
only operands are coordinates, which are 256 of 256 legal — there is nothing to
waste *on*. The composed corpus carries `REPEATX` and `XFORM`, so it has `count`,
`delta` and above all **`xf`, 8 legal values of 256 — the narrowest field in the
ISA**. If any field wastes anywhere, it is that one. **If even `xf` wastes ~0,
the "the model has not learned the grid" hypothesis is dead on every corpus this
project has**, and no architectural prior over the byte stream can recover
anything.

**B4 — the slot curve. Open.** Operand *k* is predicted after operands 0..k−1 of
its own instruction. A falling curve is within-instruction locality the model
already exploits; a flat one is locality left on the table that the waste column
cannot see, and would be a target for §7.2's proposal that survives B1–B3.

**B5 — the planner against its matched flat baseline, per field.** Claim 3's
38.75–48.97-bit loss has to live somewhere, and no instrument has said where.
Prediction: it concentrates in coordinates rather than opcodes, because
`dm/eval/redundancy.py` already measured that the summary pins the structural
bytes to within 0.035 bits/drawing. **Not a claim about the gap's cause** — the
two arms differ in more than the field mix — but a decomposition of it.

**B6 — typing per field, descriptively only.** The fusion axis is closed as
unresolvable: +2.71 at 12,000 steps and −2.42 at 24,000, both inside the
composed corpus's 0.4–4.7-bit seed spread. A per-field difference below that
floor **is not a resolution of the axis** and will not be quoted as one. It is
reported because a mechanism can be described where an effect cannot be
resolved, and saying which is which is the whole discipline.

**Not owed:** a training run, a conv front end (this exists to decide whether to
build one), a new corpus, or a second seed for anything the corpus floor already
swallows.

---

## 4. Measured, 2026-08-13 — the prior is worth 0.002–0.009 bits, and claim 3 gets a decomposition

Seven checkpoints, three val splits, no training, minutes on CPU. Records:
`runs/attribution_*.json`.

### 4.0 The instrument checks out against numbers it never saw

Every cut partitions every scored byte, so all of them must sum to the same
total — they agree to **10⁻¹² bits/drawing** on all seven arms. And that total
must be the arm's own cost, computed by an entirely different code path
(`dm/eval/metrics.py`, through a bucketing loader):

| arm | attribution | the record's own |
|---|---:|---:|
| `planbase24000eps2` flat | 556.84 | `bits_per_drawing` **556.841** |
| `plannerar24000eps2` stroke half | 414.98 | `stroke_bits` **414.984** |
| `plannerdiff24000eps2` stroke half | 416.14 | `stroke_bits` **416.136** |

Three independent reconciliations against three numbers this module cannot see.
That is what licenses everything below.

### 4.1 B1 and B2 — the waste column, and §7.2 is answered

Bits/drawing spent on symbols that **cannot occur at that position** — the exact
upper bound on what an architecture supplying the ISA grid could recover:

| corpus | arm | total budget | total waste | waste as a share |
|---|---|---:|---:|---:|
| QuickDraw ×5 | `byte` | 422.09 | **0.0019** | 1 part in 222,000 |
| QuickDraw ×5 | `token` | 423.69 | **0.0061** | 1 part in 69,000 |
| QuickDraw ×5 | `bit` | 433.68 | **0.0008** | 1 part in 542,000 |
| composed | `byte` | 459.97 | **0.0085** | 1 part in 54,000 |
| composed | `token` | 458.58 | **0.0069** | — |
| composed | `token_typed` | 457.37 | **0.0076** | — |
| QuickDraw eps2 | flat `byte` | 556.84 | **0.0028** | — |
| QuickDraw eps2 | planner-AR stroke | 414.98 | **0.0013** | — |

> **§7.2's convolutional front end has a ceiling of ~0.01 bits/drawing, on every
> corpus and every alphabet measured.** The section predicted it would "buy about
> what typing buys, which is little". It buys four orders of magnitude less than
> that: typing moves ±0.7–11.6 bits and this moves 0.002–0.009. **A model given
> 8-bit bytes and absolute positions has already learned the ISA's grid to within
> one part in fifty thousand of its budget**, and there is nothing there for an
> inductive bias to hand it.

**B2 is a null in an interesting direction.** `byte` must spread mass over 256
values where only 14 opcodes are legal, and `token` is handed the opcode region
by its alphabet — so `byte` should waste more and it **wastes less**
(0.0019 against 0.0061 on QuickDraw; 0.0085 against 0.0069 on composed, the
other way). The differences are ~0.005 bits/drawing against corpus floors of
2.5 and 0.4–4.7 bits, i.e. three orders of magnitude under the floor. **The
alphabets do not differ in what they waste, because none of them wastes
anything.** That is the mechanism behind the fusion axis's measured null, stated
where the aggregate could not state it: typing has nothing structural left to
give because the structure was never being paid for.

**B3 is dead, and it was the branch with the most room to fire.** `Kind.XF` is
the narrowest field in the ISA — **8 legal values of 256**, and no alphabet in
this project narrows it (`token_typed` gives every `Kind` a full 256-value
sub-alphabet, not its legal range). The composed `byte` arm wastes **0.0049
bits/drawing** there. So even where the ISA is 32× narrower than the alphabet
and the model was told nothing, it has found the constraint. **"The model has
not learned the grid" is refuted on every corpus this project has.**

### 4.2 B4 — within-instruction locality is already exploited

Bits per byte by operand slot, every arm:

| arm | `op` | operand 0 (x) | operand 1 (y) |
|---|---:|---:|---:|
| QuickDraw `byte` | 0.332 | 5.741 | **5.371** |
| QuickDraw `token` | 0.338 | 5.758 | **5.390** |
| QuickDraw `bit` | 0.369 | 5.852 | **5.535** |
| composed `byte` | 0.402 | 5.450 | **5.013** |
| flat eps2 `byte` | 0.320 | 5.229 | **4.892** |
| planner-AR stroke | 0.003 | 3.999 | **3.790** |

**y is cheaper than x in 6 of 6 arms**, by 0.21–0.44 bits/byte, and the gap
survives every alphabet and both corpora. y is predicted after x *of the same
instruction* and it uses it. The curve falls; the locality §7.2 wants to supply
architecturally is being exploited already.

Two caveats kept beside it. The gap is partly a property of the *data* —
QuickDraw sketches are wider than they are tall after `_fit_to_canvas`, so y is
lower-entropy before any model sees it — and this instrument cannot separate
"the model uses x" from "y was easier". What it does establish is the direction
and that no arm gets it wrong.

### 4.3 The `bit` arm has found the byte grid exactly, and the curve says so

Bits per symbol by bitplane, MSB first, `quickdraw_m5b24000eps4_bit_square_s0`:

| plane | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `opcode` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.314** | **0.055** |
| `coord_x` | 0.333 | 0.427 | 0.582 | 0.765 | 0.832 | 0.951 | 0.978 | 0.984 |
| `coord_y` | 0.293 | 0.304 | 0.513 | 0.698 | 0.825 | 0.938 | 0.976 | 0.987 |

**The bit arm spends exactly 0.000 bits on the top six bits of every opcode
byte.** QuickDraw uses `HALT`, `MOVE` and `LINE` — 0x00, 0x01, 0x02 — so those
six bits are always zero, and the model has to know *both* where a byte starts
*and* that this particular byte is an opcode to be certain of them. It is
certain. All the opcode entropy sits in the two low planes, exactly where it
lives.

And the coordinate curve is a clean monotone ramp from **0.29 to 0.99**: the
high bits are cheap because the model knows roughly where the stroke is going,
and the low bits cost a full bit each because fine position is not predictable
from anything. **The bit arm's whole excess over `byte` (+11.6 bits/drawing) is
not structure it failed to find; it is the same irreducible detail, priced one
bit at a time.**

This is the strongest available refutation of the premise behind an
architectural prior over the byte stream — and it comes from the alphabet where
the premise was most plausible, since the `bit` model is the only one that was
never told where a byte begins.

### 4.4 B5 — claim 3's loss, decomposed, and it closes against the record

Planner-AR against its matched flat baseline, identical val split
(`a54035e474876c1a`), identical bytes (160.8 per drawing). **This is the stroke
half only** — the composition level's price is not in the table and is added
below from the record.

| | planner-AR | flat | Δ |
|---|---:|---:|---:|
| `opcode` | 0.14 | 17.36 | **−17.21** |
| `coord_x` | 212.99 | 278.71 | **−65.72** |
| `coord_y` | 201.85 | 260.77 | **−58.92** |
| **stroke half** | **414.98** | **556.84** | **−141.86** |
| composition level | +181.68 | — | **+181.68** |
| **total** | **596.66** | **556.84** | **+39.82** |

The last line is claim 3's headline on record, to two decimal places, arrived at
from a decomposition that never saw it. So:

> **The factorisation wins 141.9 bits on the stroke half and pays 181.7 for the
> plan. The 39.8-bit loss is the difference between two much larger numbers,
> and neither of them had ever been reported.**

By instruction, the win is not where it looks:

| instruction | bytes/drawing | planner | flat | Δ | Δ per byte |
|---|---:|---:|---:|---:|---:|
| `MOVE` | 18.9 | 0.28 | 92.35 | **−92.07** | −4.87 |
| `LINE` | 141.0 | 414.70 | 461.82 | −47.11 | −0.33 |
| `HALT` | 1.0 | 0.00 | 2.67 | −2.67 | −2.67 |

**Two thirds of the conditioning's entire value is the `MOVE`**, which costs the
planner 0.015 bits per byte against the flat arm's 4.886 — a factor of 326. That
is `dm/eval/redundancy.py`'s `first_point` rule appearing in a decomposition
that uses no feasibility mask at all: the summary states `(x0, y0)`, so the
stroke's opening move is free. On the `LINE` bytes that make up 88% of the
program the conditioning buys a **10% discount** and no more.

> **Conditioning buys placement, not shape.** The summary tells the decoder
> where a stroke starts and how far it extends, and the decoder cashes almost
> all of that at the first instruction. What it cannot buy is the drawing.

The diffusion arm reads the same everywhere (0.13 / 212.46 / 203.54, stroke
416.14) and its 1.15-bit stroke-half deficit against the AR composition is well
under the planner's 4.16-bit replicate floor — consistency, not a result.

### 4.5 B6 — typing per field, and it stays descriptive

On the composed corpus, the one where `token` and `token_typed` are not
byte-identical, at 24,000 steps and one seed:

| field | `token` | `token_typed` | Δ |
|---|---:|---:|---:|
| `opcode` | 17.56 | 17.33 | −0.23 |
| `coord_x` | 228.97 | 228.56 | −0.41 |
| `coord_y` | 209.78 | 209.19 | −0.59 |
| `count` | 0.61 | 0.61 | −0.00 |
| `xf` | 1.66 | **1.70** | **+0.04** |
| total | 458.58 | 457.37 | −1.21 |

**Descriptive only, and the rule is the one that closed the axis.** The composed
corpus's resolution floor is 0.4–4.7 bits over eight same-config seed pairs, so
a 1.21-bit total and 0.23–0.59-bit per-field differences are inside it at one
seed. Nothing here reopens fusion.

What is worth *noting* rather than claiming: the one field where typing is
supposed to help most is the one where it reads worst. `Kind.XF` is 8 legal
values of 256, so a typed alphabet gives it a private 256-symbol region that is
97% dead — and the typed arm pays **more** there, not less. The typed axis
inflates an embedding table to separate alphabets a model had already separated
for itself (§4.1), which is a mechanism for a null that was measured two months
ago and never explained.

### 4.6 What this licenses, and what it does not

**Licensed.** "An architectural prior supplying the ISA's field grid has a
ceiling of ~0.01 bits/drawing on every corpus and alphabet measured here, so
`docs/direction.md` §7.2's convolutional front end is not worth its days."
"Claim 3's 39.8-bit loss is +181.7 for the plan against −141.9 on the strokes."
"Conditioning's value is concentrated in stroke placement: 65% of it is the
`MOVE`."

**Not licensed**, each for a named reason:

- **Not "a conv front end cannot help".** This bounds one mechanism — mass on
  symbols that cannot occur. A convolution could still help by *sharing
  parameters* across positions, which is a sample-efficiency claim this
  measures nothing about. What is refuted is the mechanism §7.2 named.
- **Not a statement about y being easier because of x.** §4.2 measures that y is
  cheaper and cannot separate the model's use of x from the corpus's own
  anisotropy.
- **Not a claim-3 verdict.** The decomposition explains the gap; it does not
  change it. The composition level's 181.7 bits are exact and at their
  asymptote, so "make the planner cheaper" still means "make the plan cheaper",
  which is the same conclusion `docs/direction.md` §3.4 reached by another road.
- **Not a fusion result.** §4.5 is inside the corpus's own floor at one seed,
  and it says so.
- **Not cross-codec per-symbol anything.** Only bits per drawing and per byte
  cross an alphabet; the `bit` arm's per-symbol column is a different
  denominator.
