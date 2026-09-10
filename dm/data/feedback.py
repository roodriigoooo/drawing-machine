"""Direction 3's paired corpora: relational, and continuation-relation-destroyed.

The pilot's primary quantity is `I_structure = G_relational - G_destroyed`. The
control is a continuation-relation destruction with exact listed marginals. It
preserves the byte-level quantities a whole-block permutation can preserve, while
changing seam bigrams and local continuity by design. This is narrower than a
claim that relation and nothing else changed at every token-level scale:

| property | relational | destroyed |
|---|---|---|
| program count, order and byte length | — | identical |
| opcode/operand skeleton, per program | — | identical |
| corpus-wide byte multiset | — | identical |
| corpus-wide continuation-block multiset | — | identical |
| which continuation follows which prefix | the step implies it | a stranger's |
| seam bigrams and local continuity | original | intentionally changed |

That table is why the intervention is a **permutation** rather than a
regeneration or a perturbation. Regenerating with a different rule changes the
marginals; perturbing bytes changes the n-gram statistics inside the block.
Moving whole blocks between programs gives an auditable intervention with exact
listed marginals and a measured seam change; the program-level treatment rate and
full-corpus VM fault/stroke census are published alongside it.

**Model-blind, and that is a rule rather than a convenience.** Nothing here loads
a checkpoint, reads a logit or consults a score. The donor map is a function of
the corpus, the frozen strata and one seed
(`docs/directions.md` §7 invariant 1). A seed is never retried because a balance
number or a later model score looks inconvenient.

**Provenance comes from the generator, not from an oracle.** `dm.eval.recovery`
recovers repeats from a flat program by folding it, which is a good instrument
and an inference. Here the L1 program is kept beside its flat trace, so the copy
spans are known rather than deduced -- and the arithmetic is checked against the
flat bytes it claims to describe, so a wrong offset fails loudly instead of
deranging the wrong region.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..isa.asm import parse
from ..isa.codec import Codec
from ..isa.spec import SPECS, Op, Tier
from ..isa.unroll import _chunks, _matching, unroll
from .augment import Affine, apply
from .feedback_audit import audit_accepts, vm_census
from .fingerprint import fingerprint
from .synthetic import Unrollable, sample

#: Bumped whenever the byte-level balance contract changes. A balance repair
#: bumps it and rebuilds -- which is only cheap while no weights exist, and is
#: the reason F4 precedes F5.
CORPUS_SCHEMA = 1

#: Manifest schema is separate from the balance schema because adding audit
#: evidence changes the frozen manifest interface even when the paired bytes
#: stay identical. The old schema-1 manifest remains valid historical bytes but
#: cannot authorize a new training freeze.
MANIFEST_SCHEMA = 2

#: The first ordinal whose block is a *continuation*: copies 1 and 2 establish
#: the step and copy 3 onward is what the step implies. Direction 2's step venue
#: scores exactly ordinal 3, and every later copy stands in the same relation to
#: the same two, so the whole tail is deranged. Leaving ordinal 4 in place would
#: leave a longer-range copy of the relation the control exists to remove.
FIRST_CONTINUATION_ORDINAL = 3

#: Scope opcodes this corpus is allowed to contain. The flat synthetic corpus is
#: built from `random_grid`, whose only scope is a non-nested `REPEAT`; anything
#: else means the generator changed under the builder and the span arithmetic
#: below no longer describes it.
_SCOPE_OPCODES = frozenset({"REPEAT", "REPEATX", "XFORM", "ENDREP", "ENDX"})

Signature = tuple[tuple[str, tuple[str, ...]], ...]


def skeleton(program: bytes) -> Signature:
    """The opcode/operand-kind shape, which is what "same block" means here.

    Equal byte length is not enough: `MOVE x y` and `CIRCLE r` followed by `FILL`
    occupy the same three bytes and are not interchangeable blocks.

    Computed here rather than imported from `dm.eval.context.signature`, which
    does the same thing: `dm.data` must not depend on `dm.eval`, and a corpus
    builder that reached into the evaluation package would be one import away
    from a checkpoint.
    """
    return tuple(
        (instr.mnemonic,
         tuple(kind.name for kind in SPECS[Op[instr.mnemonic]].operands))
        for instr in parse(program)
    )


# ---------------------------------------------------------------------------
# generator provenance


@dataclass(frozen=True)
class RepeatScope:
    """One expanded `REPEAT`, and where its copies land in the flat trace."""

    start: int
    count: int
    dx: int
    dy: int
    body_bytes: int

    def span(self, ordinal: int) -> tuple[int, int]:
        """Byte range of the 1-based `ordinal`-th copy."""
        if not 1 <= ordinal <= self.count:
            raise ValueError(f"ordinal {ordinal} outside 1..{self.count}")
        begin = self.start + (ordinal - 1) * self.body_bytes
        return begin, begin + self.body_bytes


def repeat_scopes(l1: bytes) -> list[RepeatScope]:
    """Where each top-level `REPEAT` in `l1` writes its copies in the flat trace.

    Offsets are computed by walking the L1 instruction stream: a plain
    instruction contributes its own bytes and a `REPEAT count dx dy` contributes
    `count` bodies. `verify_scopes` then checks the answer against the flat bytes,
    so an arithmetic slip cannot quietly point the derangement at the wrong
    region.
    """
    chunks = _chunks(l1)
    scopes: list[RepeatScope] = []
    offset = 0
    index = 0
    while index < len(chunks):
        instr, raw = chunks[index]
        if instr.mnemonic != "REPEAT":
            if instr.mnemonic in _SCOPE_OPCODES:
                raise ValueError(
                    f"unexpected scope opcode {instr.mnemonic!r}: this corpus is "
                    "built from non-nested REPEAT only"
                )
            offset += len(raw)
            index += 1
            continue
        end = _matching(chunks, index)
        if end is None:
            raise ValueError("unterminated REPEAT in a program that unrolled")
        body_chunks = chunks[index + 1 : end - 1]
        if any(chunk[0].mnemonic in _SCOPE_OPCODES for chunk in body_chunks):
            raise ValueError("nested scope: the span arithmetic assumes a flat body")
        body_bytes = sum(len(chunk[1]) for chunk in body_chunks)
        count, dx, dy = instr.args
        scopes.append(RepeatScope(start=offset, count=count, dx=dx, dy=dy,
                                  body_bytes=body_bytes))
        offset += count * body_bytes
        index = end
    return scopes


def verify_scopes(flat: bytes, scopes: list[RepeatScope]) -> None:
    """Assert every claimed copy really is the first copy, translated.

    The whole control rests on cutting at exactly the right byte offsets. This
    turns an off-by-one from a silent corpus defect into a build failure.
    """
    for scope in scopes:
        first_start, first_stop = scope.span(1)
        body = flat[first_start:first_stop]
        for ordinal in range(2, scope.count + 1):
            start, stop = scope.span(ordinal)
            step = Affine(dx=(ordinal - 1) * scope.dx, dy=(ordinal - 1) * scope.dy)
            if flat[start:stop] != apply(body, step):
                raise ValueError(
                    f"copy {ordinal} at {start}:{stop} is not copy 1 translated by "
                    f"{(ordinal - 1) * scope.dx, (ordinal - 1) * scope.dy}"
                )


@dataclass(frozen=True)
class SourceProgram:
    """A flat training program beside the L1 program that produced it."""

    flat: bytes
    l1: bytes
    scopes: tuple[RepeatScope, ...]


def generate_with_provenance(n: int, *, seed: int, tier: Tier = Tier.L1,
                             exclude: frozenset[bytes] = frozenset(),
                             **kwargs) -> list[SourceProgram]:
    """`synthetic.dataset(n, seed, flatten=True)`, with the L1 source retained.

    A faithful re-run of that loop rather than a second sampler: the same
    generator, the same rejection of programs with no exact flat trace, the same
    deduplication on the *flat* program, the same rejection bound. The only
    difference is that the L1 program is kept, which is the whole point --
    `dm.data.synthetic` throws it away and the copy spans with it.

    `test_the_provenance_stream_is_the_ordinary_corpus` pins the equality, so a
    change to the generator cannot leave this quietly reproducing an old corpus.
    """
    # Accepted and dropped rather than forwarded: flattening is what this builder
    # does, and the key is present in a `TrainConfig.extra` that names the corpus
    # for every other Tier A caller. Refusing `flatten=False` keeps that key
    # meaningful instead of silently ignored.
    if not kwargs.pop("flatten", True):
        raise ValueError(
            "the feedback corpus is the flat trace by construction; the step "
            "relation only exists once REPEAT has been expanded"
        )
    rng = random.Random(seed)
    seen: set[bytes] = set(exclude)
    out: list[SourceProgram] = []
    for _ in range(100 * n + 1000):
        l1 = sample(rng, tier=tier, flatten=False, **kwargs)
        flat = unroll(l1)
        if flat is None:
            continue  # `sample(flatten=True)` raises Unrollable here
        if flat in seen:
            continue
        seen.add(flat)
        scopes = repeat_scopes(l1)
        verify_scopes(flat, scopes)
        out.append(SourceProgram(flat=flat, l1=l1, scopes=tuple(scopes)))
        if len(out) == n:
            return out
    raise Unrollable(
        f"only {len(out)}/{n} distinct programs at seed {seed}; the motif entropy "
        "is too low for a split this size"
    )


# ---------------------------------------------------------------------------
# continuation blocks and their strata


@dataclass(frozen=True)
class Block:
    """One continuation copy: the bytes a step relation implies."""

    program: int
    scope: int
    ordinal: int
    start: int
    stop: int
    content: bytes
    #: What the source program's own relation implies at this position. Equal to
    #: `content` in the relational arm by construction; a donor equal to it would
    #: preserve the very relation this control removes.
    implied: bytes

    @property
    def stratum(self) -> tuple[int, Signature, int]:
        """Byte length, skeleton and target ordinal, as §5.1 freezes them."""
        return (self.stop - self.start, skeleton(self.content), self.ordinal)


def continuation_blocks(programs: list[SourceProgram]) -> list[Block]:
    """Every copy at ordinal 3 or later, across the whole training split."""
    blocks: list[Block] = []
    for program_index, source in enumerate(programs):
        for scope_index, scope in enumerate(source.scopes):
            first_start, first_stop = scope.span(1)
            body = source.flat[first_start:first_stop]
            for ordinal in range(FIRST_CONTINUATION_ORDINAL, scope.count + 1):
                start, stop = scope.span(ordinal)
                implied = apply(body, Affine(dx=(ordinal - 1) * scope.dx,
                                             dy=(ordinal - 1) * scope.dy))
                blocks.append(Block(
                    program=program_index, scope=scope_index, ordinal=ordinal,
                    start=start, stop=stop, content=source.flat[start:stop],
                    implied=implied if implied is not None else b"",
                ))
    return blocks


# ---------------------------------------------------------------------------
# the derangement


class Unbalanced(ValueError):
    """A stratum admits no complete derangement.

    Raised rather than dropping the stratum. Dropping would shorten the destroyed
    arm's programs, and then the two arms would differ in length as well as in
    the relation -- which is exactly the confound the whole construction exists to
    avoid (§5.1 rule 5).
    """


def derange(blocks: list[Block], *, seed: int) -> tuple[dict[int, int], Counter]:
    """Map every block index to a donor's, model-blind and deterministically.

    Within a stratum the indices are shuffled once and then **cycled by one**.
    That gives a permutation with no fixed point by construction, at any stratum
    size down to two, and it hands every block exactly one donation -- so the
    corpus-wide block multiset is preserved exactly and donor reuse is uniform.
    Rejection-sampling a derangement instead would leave the reuse distribution
    to chance and the runtime to luck.

    One assignment is still forbidden after that: a donor whose bytes equal what
    the source's own relation implies. It is rare -- it needs a stranger's block
    to coincide byte for byte with the step's own continuation -- and it is
    repaired by the first swap that makes both positions legal, scanning in a
    fixed order so the repair is a function of the corpus and the seed.
    """
    census: Counter = Counter()
    strata: dict[tuple, list[int]] = {}
    for index, block in enumerate(blocks):
        strata.setdefault(block.stratum, []).append(index)

    donors: dict[int, int] = {}
    for key in sorted(strata, key=lambda k: (k[0], k[2], str(k[1]))):
        members = strata[key]
        census["strata"] += 1
        if len(members) < 2:
            census["stratum_too_small"] += 1
            raise Unbalanced(
                f"stratum {key[0]} bytes / ordinal {key[2]} has {len(members)} "
                "block(s) and cannot be deranged; a stratum is never dropped"
            )
        order = list(members)
        random.Random(f"{seed}:{key[0]}:{key[2]}:{key[1]}").shuffle(order)
        assigned = {order[i]: order[(i + 1) % len(order)]
                    for i in range(len(order))}
        _repair(blocks, order, assigned, census)
        donors.update(assigned)
        census["blocks"] += len(members)
    return donors, census


def _forbidden(blocks: list[Block], target: int, donor: int) -> str | None:
    if target == donor:
        return "identity_donor"
    if blocks[donor].content == blocks[target].implied:
        return "relation_preserving_donor"
    return None


def _repair(blocks: list[Block], order: list[int], assigned: dict[int, int],
            census: Counter) -> None:
    """Swap out donors that would preserve the relation, in a fixed scan order."""
    for position, target in enumerate(order):
        reason = _forbidden(blocks, target, assigned[target])
        if reason is None:
            continue
        census[f"rejected_{reason}"] += 1
        for offset in range(1, len(order)):
            other = order[(position + offset) % len(order)]
            mine, theirs = assigned[target], assigned[other]
            if (_forbidden(blocks, target, theirs) is None
                    and _forbidden(blocks, other, mine) is None):
                assigned[target], assigned[other] = theirs, mine
                census["repaired_by_swap"] += 1
                break
        else:
            raise Unbalanced(
                f"block {target} has no legal donor in its stratum; every "
                "candidate either is itself or reproduces its own relation"
            )


def splice(programs: list[SourceProgram], blocks: list[Block],
           donors: dict[int, int]) -> list[bytes]:
    """The destroyed arm: every continuation block replaced by its donor's bytes.

    Whole blocks, so the n-grams *inside* a continuation are untouched and only
    the splice changes. Applied back to front within a program so earlier offsets
    stay valid, which also means the offsets never need recomputing and cannot
    drift.
    """
    per_program: dict[int, list[Block]] = {}
    for index, block in enumerate(blocks):
        per_program.setdefault(block.program, []).append(
            Block(**{**block.__dict__, "content": blocks[donors[index]].content})
        )
    out: list[bytes] = []
    for program_index, source in enumerate(programs):
        payload = bytearray(source.flat)
        for block in sorted(per_program.get(program_index, ()),
                            key=lambda b: b.start, reverse=True):
            payload[block.start:block.stop] = block.content
        if len(payload) != len(source.flat):
            raise Unbalanced(
                f"program {program_index} changed length under splicing; a donor "
                "of a different byte length reached a stratum"
            )
        out.append(bytes(payload))
    return out


# ---------------------------------------------------------------------------
# the paired corpus and its census


@dataclass(frozen=True)
class PairedCorpus:
    """Both training arms, one shared validation split, and the whole census."""

    relational: list[bytes]
    destroyed: list[bytes]
    val: list[bytes]
    blocks: list[Block] = field(repr=False, default_factory=list)
    donors: dict[int, int] = field(repr=False, default_factory=dict)
    census: Counter = field(repr=False, default_factory=Counter)
    seed: int = 0
    data_seed: int = 0


def build(n_train: int, n_val: int, *, data_seed: int = 0, seed: int,
          tier: Tier = Tier.L1, **kwargs) -> PairedCorpus:
    """Both arms from one source, with only the splice differing.

    The validation split is byte-identical between arms and always relational:
    the two arms are compared on the same held-out programs, and a control arm
    validated on its own control distribution would be measuring how well it fits
    its own intervention.
    """
    train = generate_with_provenance(n_train, seed=data_seed, tier=tier, **kwargs)
    relational = [source.flat for source in train]
    val = generate_with_provenance(
        n_val, seed=data_seed + 10_000, tier=tier,
        exclude=frozenset(relational), **kwargs,
    )
    blocks = continuation_blocks(train)
    if not blocks:
        raise Unbalanced(
            "no continuation blocks: this corpus has no repeat with three copies, "
            "so there is no relation to destroy"
        )
    donors, census = derange(blocks, seed=seed)
    destroyed = splice(train, blocks, donors)
    return PairedCorpus(
        relational=relational,
        destroyed=destroyed,
        val=[source.flat for source in val],
        blocks=blocks, donors=donors, census=census,
        seed=seed, data_seed=data_seed,
    )


def _coordinate_marginals(programs: list[bytes]) -> dict[str, dict[int, int]]:
    """Operand-value counts per operand kind, over the whole split.

    A permutation of whole blocks cannot change these, so the two arms have to
    agree exactly. Reported as an equality rather than a tolerance for that
    reason: a difference here means a donor of the wrong shape got through.
    """
    counts: dict[str, Counter] = {}
    for program in programs:
        pc = 0
        for instr in parse(program):
            spec = SPECS[Op[instr.mnemonic]]
            for offset, kind in enumerate(spec.operands, start=1):
                counts.setdefault(kind.name, Counter())[program[pc + offset]] += 1
            pc += spec.size
    return {name: dict(sorted(counter.items())) for name, counter in counts.items()}


def _seam_bigrams(programs: list[bytes], blocks: list[Block]) -> dict[str, int]:
    """The byte pair straddling each splice, counted.

    The one distribution the intervention is *supposed* to move. It is reported
    rather than balanced: a control whose seam statistics matched the relational
    arm exactly would not have changed the splice, which is the intervention.
    """
    counts: Counter = Counter()
    for block in blocks:
        if block.start == 0:
            continue
        program = programs[block.program]
        counts[f"{program[block.start - 1]:02x}{program[block.start]:02x}"] += 1
    return dict(sorted(counts.items()))


def _total_variation(left: dict[str, int], right: dict[str, int]) -> float:
    total_left = sum(left.values()) or 1
    total_right = sum(right.values()) or 1
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / total_left - right.get(key, 0) / total_right)
        for key in keys
    )


def balance_report(corpus: PairedCorpus, codec: Codec | None = None) -> dict:
    """Everything a reviewer needs before the arms are allowed to train.

    Four exact equalities and one reported difference. The equalities are exact
    rather than toleranced because a permutation of whole blocks *cannot* move
    them -- program lengths, skeletons, the corpus byte multiset and the
    coordinate marginals are all invariant under it -- so any drift is a defect
    and not noise. The seam bigram is the intervention itself, and it has to have
    bitten: a control whose splices matched the relational arm's would be a
    control in name only.
    """
    relational, destroyed = corpus.relational, corpus.destroyed
    lengths_match = [len(a) == len(b) for a, b in zip(relational, destroyed)]
    skeletons_match = [skeleton(a) == skeleton(b)
                       for a, b in zip(relational, destroyed)]
    bytes_relational = Counter(byte for p in relational for byte in p)
    bytes_destroyed = Counter(byte for p in destroyed for byte in p)
    marginals_relational = _coordinate_marginals(relational)
    marginals_destroyed = _coordinate_marginals(destroyed)
    seam_relational = _seam_bigrams(relational, corpus.blocks)
    seam_destroyed = _seam_bigrams(destroyed, corpus.blocks)
    moved = sum(1 for index, block in enumerate(corpus.blocks)
                if corpus.blocks[corpus.donors[index]].content != block.content)
    reuse = Counter(corpus.donors.values())
    treated_programs = {block.program for block in corpus.blocks}
    program_treatment = {
        "treated_programs": len(treated_programs),
        "untreated_programs": len(relational) - len(treated_programs),
        "treated_fraction": len(treated_programs) / max(1, len(relational)),
    }
    return {
        "schema": CORPUS_SCHEMA,
        "n_train": len(relational),
        "n_val": len(corpus.val),
        "blocks": len(corpus.blocks),
        "strata": corpus.census.get("strata", 0),
        "census": dict(sorted(corpus.census.items())),
        "exact": {
            "program_lengths_preserved": all(lengths_match),
            "skeletons_preserved": all(skeletons_match),
            "byte_multiset_preserved": bytes_relational == bytes_destroyed,
            "coordinate_marginals_preserved":
                marginals_relational == marginals_destroyed,
            "no_identity_donor": all(corpus.donors[i] != i
                                     for i in range(len(corpus.blocks))),
            "no_relation_preserving_donor": all(
                corpus.blocks[corpus.donors[i]].content != block.implied
                for i, block in enumerate(corpus.blocks)
            ),
            "donor_reuse_is_uniform": set(reuse.values()) <= {1},
            # Splicing can in principle land a destroyed program on a held-out
            # one. Val is drawn disjoint from the *relational* arm, so the
            # destroyed arm needs its own check or the control arm alone would be
            # scoring recall as generalisation.
            "destroyed_train_val_disjoint":
                not (set(destroyed) & set(corpus.val)),
            "relational_train_val_disjoint":
                not (set(relational) & set(corpus.val)),
        },
        "intervention": {
            # A block whose donor happens to carry identical bytes is a real
            # outcome of a permutation over a finite multiset, not a failure --
            # what it must not be is common.
            "blocks_whose_content_changed": moved,
            "content_change_rate": moved / max(1, len(corpus.blocks)),
            "seam_bigram_total_variation": _total_variation(seam_relational,
                                                            seam_destroyed),
            "program_treatment": program_treatment,
        },
        "ordinals": dict(sorted(Counter(b.ordinal for b in corpus.blocks).items())),
        "block_bytes": dict(sorted(Counter(b.stop - b.start
                                           for b in corpus.blocks).items())),
        "program_bytes": {
            "relational_total": sum(len(p) for p in relational),
            "destroyed_total": sum(len(p) for p in destroyed),
        },
        "corpus": {
            "relational": fingerprint(relational, corpus.val),
            "destroyed": fingerprint(destroyed, corpus.val),
        },
        "seed": corpus.seed,
        "data_seed": corpus.data_seed,
        "codec": None if codec is None else codec.name,
    }


def donor_examples(corpus: PairedCorpus, limit: int = 8) -> list[dict]:
    """A human-readable sample of splices, for the review §6 F4 requires.

    A fingerprint is accepted by a person looking at real donors, not by a
    checker confirming its own arithmetic. Every field is hex so the reviewer
    sees bytes rather than a rendering that might be hiding the defect.
    """
    out = []
    for index, block in enumerate(corpus.blocks[:limit]):
        donor = corpus.blocks[corpus.donors[index]]
        out.append({
            "block": index,
            "program": block.program,
            "ordinal": block.ordinal,
            "span": [block.start, block.stop],
            "own_content": block.content.hex(),
            "implied_by_relation": block.implied.hex(),
            "donor_program": donor.program,
            "donor_content": donor.content.hex(),
            "changed": donor.content != block.content,
        })
    return out


def manifest(corpus: PairedCorpus, codec: Codec | None = None) -> dict:
    """The frozen record of a build, content-addressed like every other artifact.

    Carries the donor map in full. A census without the map cannot be re-checked,
    and §5.1 asks for both because the map is what an auditor replays.
    """
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen",
        "balance": balance_report(corpus, codec),
        "donor_examples": donor_examples(corpus),
        "donor_map": [corpus.donors[i] for i in range(len(corpus.blocks))],
        "blocks": [
            [b.program, b.scope, b.ordinal, b.start, b.stop]
            for b in corpus.blocks
        ],
        "vm_census": vm_census({
            "relational": corpus.relational,
            "destroyed": corpus.destroyed,
            "validation": corpus.val,
        }),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["corpus_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def write_manifest(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def manifest_digests(path: Path) -> dict:
    """Both hashes of a manifest, under names that cannot be swapped.

    They identify **different byte strings**. The canonical payload digest is
    SHA-256 over the sorted, separator-normalised JSON body with the recorded
    digest removed; the file hash is SHA-256 over the bytes on disk, which also
    cover indentation, key order as written and the trailing newline. Reformat
    the file and the first is unchanged while the second moves.

    `PLAN.md` names calling one the other as a trap, and it was a real one: the
    audited corpus's canonical digest is `8da5b8fb...` and its file hash is
    `08a53e3b...`. Returning both from one function, keyed by what they are, is
    what stops a caller quoting whichever it happens to hold.
    """
    body = json.loads(path.read_text())
    payload = {key: value for key, value in body.items() if key != "corpus_sha256"}
    canonical = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "canonical_payload_sha256": canonical,
        "recorded_canonical_sha256": body.get("corpus_sha256"),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_manifest(path: Path) -> dict:
    body = json.loads(path.read_text())
    recorded = body.get("corpus_sha256")
    payload = {k: v for k, v in body.items() if k != "corpus_sha256"}
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded != actual:
        raise ValueError(f"feedback corpus manifest hash mismatch for {path}")
    if body.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported feedback corpus schema {body.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA}"
        )
    if body.get("status") != "frozen":
        raise ValueError(f"feedback corpus manifest is not frozen: {path}")
    if "vm_census" not in body:
        raise ValueError(
            f"feedback corpus manifest has no full-corpus VM census for {path}: "
            "incomplete"
        )
    if not audit_accepts(body["vm_census"]):
        raise ValueError(
            f"feedback corpus manifest has an incomplete, faulted or unbalanced "
            f"VM census for {path}: every arm must be all-valid, all-halted and "
            f"zero-fault, and the two training arms must agree: incomplete"
        )
    treatment = (body.get("balance") or {}).get("intervention", {}).get(
        "program_treatment"
    )
    if not isinstance(treatment, dict):
        raise ValueError(  # noqa: TRY004 -- malformed artifacts fail closed uniformly
            f"feedback corpus manifest has no program-level treatment prevalence "
            f"for {path}: incomplete"
        )
    return body


def accepts(report: dict) -> bool:
    """Whether a balance report clears every frozen condition.

    All five exact invariants, an intervention that actually bit, and a program
    treatment prevalence that **reconciles**: `treated + untreated == n_train`
    was already checked and the reported fraction was not, so a report could
    quote a prevalence none of its own counts produces. Returned as a boolean
    over a report a human has already read, not as a gate the builder applies to
    itself.
    """
    intervention = report.get("intervention") or {}
    treatment = intervention.get("program_treatment")
    treated = treatment.get("treated_programs") if isinstance(treatment, dict) else None
    untreated = (treatment.get("untreated_programs")
                 if isinstance(treatment, dict) else None)
    fraction = treatment.get("treated_fraction") if isinstance(treatment, dict) else None
    treatment_complete = (
        isinstance(treatment, dict)
        and isinstance(treated, int) and not isinstance(treated, bool)
        and isinstance(untreated, int) and not isinstance(untreated, bool)
        and isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
        and treated + untreated == report.get("n_train")
        and 0.0 < fraction < 1.0
        # The fraction has to be the one the counts produce. Without this, three
        # numbers can agree pairwise and not jointly, and the prevalence a
        # protocol quotes is not one any pair of its own counts supports.
        and isinstance(report.get("n_train"), int)
        and report["n_train"] > 0
        and math.isclose(float(fraction), treated / report["n_train"],
                         rel_tol=0.0, abs_tol=1e-12)
    )
    return (all(report.get("exact", {}).values())
            and intervention.get("content_change_rate", 0.0) > 0.5
            and intervention.get("seam_bigram_total_variation", 0.0) > 0.0
            and treatment_complete)


__all__ = [
    "CORPUS_SCHEMA",
    "FIRST_CONTINUATION_ORDINAL",
    "MANIFEST_SCHEMA",
    "Block",
    "PairedCorpus",
    "RepeatScope",
    "SourceProgram",
    "Unbalanced",
    "accepts",
    "balance_report",
    "build",
    "continuation_blocks",
    "derange",
    "donor_examples",
    "generate_with_provenance",
    "load_manifest",
    "manifest",
    "manifest_digests",
    "repeat_scopes",
    "skeleton",
    "splice",
    "verify_scopes",
    "write_manifest",
]
