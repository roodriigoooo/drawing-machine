"""The C port, checked against the specification it mirrors.

Three kinds of test, and only the third needs a compiler:

- **The mirror.** `port/` restates the ISA in C. A restatement is a convention
  until something checks it, and `PLAN.md` §10 already carries the cost of one
  unchecked convention per corpus mix-up. These tests parse the headers and
  assert they agree with `dm/isa/spec.py` -- including the two structural
  assumptions the interpreter *exploits*, which are true of ISA v2 and not
  guaranteed by it: opcodes are contiguous from zero, and `Kind.DELTA` occurs
  only in the three loop-and-transform instructions.
- **The exactness property.** `port/include/dm_vm.h` argues that the reference
  VM's geometry is exactly representable at a scale of `CURVE_STEPS ** 3`, which
  is the whole reason a fixed-point port can be bit-exact rather than close.
  That argument is checkable in pure Python and is checked here, so it fails
  even on a machine with no toolchain.
- **Conformance.** A short run of `scripts/conformance.py`'s comparison, skipped
  when there is no compiler. The full sweep over every corpus is the script.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dm.data import synthetic
from dm.isa.spec import CANVAS, MAX_REPEAT_DEPTH, SPECS, Kind, Op, Tier
from dm.isa.transform import D4, D4_ORDER
from dm.vm.interp import CURVE_STEPS, VM, FaultKind

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
ISA_H = (PORT / "include" / "dm_isa.h").read_text()
VM_H = (PORT / "include" / "dm_vm.h").read_text()
ISA_C = (PORT / "src" / "dm_isa.c").read_text()
ISA_VM_C = (PORT / "src" / "dm_vm.c").read_text()


def define(header: str, name: str) -> int:
    """The integer a plain `#define` expands to.

    Deliberately refuses anything that is not the whole macro body: an earlier
    version matched a leading integer, which quietly read `DM_FRAC_BITS` as 3
    from `(3 * DM_CURVE_STEPS_LOG2)` and would have compared a constant against
    a fragment of itself. A parser that half-succeeds on a mirror test is worse
    than no mirror test.
    """
    match = re.search(rf"^#define {name}\s+(\S+)\s*$", header, re.MULTILINE)
    assert match, f"{name} is not defined, or is not on one line"
    body = match.group(1).strip("()")
    assert re.fullmatch(r"-?\d+", body), f"{name} expands to {body!r}, not a plain integer"
    return int(body)


# --------------------------------------------------------------------------
# the mirror


#: Opcodes the reference has and the port does not. **Empty since 2026-08-11**:
#: P4 landed the transform tier on the device, so the mirror is total again.
#:
#: The name survives its own emptiness on purpose. It is the one place a reader
#: learns whether the device's ISA is the reference's, and an assertion that the
#: gap is *exactly* this set fails when a fifteenth opcode arrives -- whether it
#: is ported or not.
NOT_PORTED: set[str] = set()


def test_opcode_values_match_spec():
    declared = dict(re.findall(r"DM_OP_(\w+)\s*=\s*0x([0-9A-Fa-f]+)", ISA_H))
    assert {name: int(value, 16) for name, value in declared.items()} == {
        op.name: int(op) for op in Op if op.name not in NOT_PORTED
    }


def test_instruction_sizes_match_spec():
    declared = dict(re.findall(r"\[DM_OP_(\w+)\]\s*=\s*(\d+)", ISA_C))
    assert {name: int(size) for name, size in declared.items()} == {
        spec.op.name: spec.size for spec in SPECS.values()
        if spec.op.name not in NOT_PORTED
    }


def test_the_unported_opcodes_are_exactly_the_ones_named():
    """The gap itself is the assertion, and it is currently empty: the device
    implements the whole reference ISA. Written as an equality rather than a
    subset so a new opcode fails here whether or not it was ported."""
    declared = set(re.findall(r"DM_OP_(\w+)\s*=", ISA_H))
    assert {op.name for op in Op} - declared == NOT_PORTED


def test_opcode_count_matches():
    assert define(ISA_H, "DM_N_OPCODES") == len(SPECS) - len(NOT_PORTED)


def test_canvas_and_depth_match():
    assert define(ISA_H, "DM_CANVAS") == CANVAS
    assert define(ISA_H, "DM_MAX_REPEAT_DEPTH") == MAX_REPEAT_DEPTH


def test_opcodes_are_contiguous_from_zero():
    """`dm_is_opcode` is `byte < DM_N_OPCODES`, which is only a complete test
    while the opcode values have no holes. A reserved gap in a future ISA would
    make the port accept a byte the reference faults on.

    Any unported tier has to sit at the *end* of the range, or `DM_N_OPCODES`
    would stop describing a contiguous prefix and `dm_is_opcode` would accept a
    byte the device cannot execute. Vacuous while everything is ported, and the
    assertion that keeps it true if that changes.
    """
    assert sorted(int(op) for op in Op) == list(range(len(SPECS)))
    ported = sorted(int(op) for op in Op if op.name not in NOT_PORTED)
    assert ported == list(range(len(ported))), "the unported tier must be a suffix"


def test_delta_operands_occur_only_where_the_port_sign_extends():
    """The port sign-extends at three named sites instead of carrying an
    operand-kind table into flash, and this is what makes that sound.

    Under ISA v1 it was one site. The transform tier widened it to three -- the
    bill this test predicted before P4, and the reason it states the exact set
    rather than a count: a fourth signed instruction fails here, where the fix
    is cheap, rather than in the geometry on a device.
    """
    signed = {spec.op for spec in SPECS.values() if Kind.DELTA in spec.operands}
    assert signed == {Op.REPEAT, Op.XFORM, Op.REPEATX}


def test_the_ports_closed_form_d4_composition_matches_the_reference():
    """`port/src/dm_vm.c` composes D4 with a formula instead of a 64-byte table:
    writing an element as `s^m r^t` and using `r^t s = s r^-t`,

        (s^m2 r^t2)(s^m1 r^t1) = s^(m1^m2) r^(t1 + (m1 ? -t2 : t2))

    The reference derives its table from the *action* on the canvas rather than
    by transcription, so agreeing on all 64 products is a real check on the
    algebra and not two copies of one typo. The formula is restated here in the
    C's own terms; conformance checks that the C implements it.
    """
    formula = re.search(r"const int32_t turns = \(m1 \? (.+?)\) & 3;", ISA_VM_C)
    assert formula, "the closed form is not where its comment says it is"
    assert formula.group(1) == "t1 - t2 : t1 + t2"

    for first in range(D4_ORDER):
        for second in range(D4_ORDER):
            m1, t1 = first & 1, (first >> 1) & 3
            m2, t2 = second & 1, (second >> 1) & 3
            got = (((t1 - t2 if m1 else t1 + t2) & 3) << 1) | (m1 ^ m2)
            assert got == D4.of(first).then(D4.of(second)).code, (first, second)


def test_the_transform_stack_is_bounded_like_the_repeat_stack():
    """Two stacks, two bounds, and the device must agree with the reference on
    both -- a program that faults on one side and runs on the other is the
    divergence conformance cannot catch, because it never gets to compare
    geometry."""
    alias = re.search(r"^#define DM_MAX_XFORM_DEPTH (\S+)\s*$", ISA_H, re.MULTILINE)
    assert alias, "DM_MAX_XFORM_DEPTH is not defined"
    # Either spelling is fine; both have to mean the reference's bound. `define`
    # deliberately refuses a macro that expands to a macro, so the alias is
    # matched rather than evaluated.
    assert (alias.group(1) == "DM_MAX_REPEAT_DEPTH"
            or int(alias.group(1)) == MAX_REPEAT_DEPTH)
    assert define(ISA_H, "DM_MAX_REPEAT_DEPTH") == MAX_REPEAT_DEPTH
    assert define(ISA_H, "DM_D4_ORDER") == D4_ORDER


def test_fault_kinds_match():
    declared = re.findall(r"DM_FAULT_(\w+)", VM_H)
    assert {name.lower() for name in declared} == {kind.value for kind in FaultKind}


def test_curve_steps_matches_reference_and_is_a_power_of_two():
    """Exactness needs a binary `t`. At a non-power-of-two the reference's own
    Bernstein coefficients stop being binary fractions and no integer port can
    reproduce them."""
    log2 = define(VM_H, "DM_CURVE_STEPS_LOG2")
    assert 1 << log2 == CURVE_STEPS
    assert CURVE_STEPS & (CURVE_STEPS - 1) == 0


def test_fixed_point_scale_is_derived_from_the_curve_count():
    """`DM_FRAC_BITS` must be *derived* from the curve count, not chosen to
    match it. A literal here would be correct today and silently wrong the first
    time `DM_CURVE_STEPS_LOG2` moved, which is precisely when flattening stops
    being exact. The compiled value is checked separately: the harness prints
    `frac_bits` and `scripts/conformance.py` refuses a port whose scale differs
    from the reference's."""
    assert re.search(r"^#define DM_FRAC_BITS \(3 \* DM_CURVE_STEPS_LOG2\)\s*$", VM_H, re.MULTILINE)
    assert 3 * define(VM_H, "DM_CURVE_STEPS_LOG2") == 3 * CURVE_STEPS.bit_length() - 3


# --------------------------------------------------------------------------
# the exactness property


@pytest.mark.parametrize("tier,depth", [(Tier.L0, 2), (Tier.L1, 2), (Tier.L1, 0)])
def test_reference_geometry_is_exact_at_the_port_scale(tier, depth):
    """Every coordinate the reference produces is an exact multiple of 1/ONE.

    This is the load-bearing claim of the whole port: it is why the device
    needs no FPU *and* loses no geometry, and why `scripts/conformance.py`
    compares with `==` rather than a tolerance. A failure here means the C is
    not wrong -- the argument is.
    """
    one = CURVE_STEPS**3
    vm = VM()
    checked = 0
    for program in synthetic.dataset(60, seed=7, tier=tier, depth=depth):
        trace = vm.run(program)
        points = [pt for s in trace.strokes for pt in s.points]
        points += [pt for r in trace.regions for pt in r.points]
        points += [(d.cx, d.cy) for d in trace.discs]
        for x, y in points:
            for value in (x, y):
                assert value * one == int(value * one), f"{value} is not a multiple of 1/{one}"
                assert 0 <= value <= CANVAS - 1
                checked += 1
    assert checked > 0


def test_curve_coefficients_sum_to_the_scale():
    """The port's flattening is `A*x0 + B*x1 + C*x2 + D*x3` with no division, so
    an affine combination only stays inside the canvas while the weights sum to
    exactly `ONE`. Checked here in the same integers the C uses."""
    for i in range(1, CURVE_STEPS + 1):
        u = CURVE_STEPS - i
        assert u**3 + 3 * u * u * i + 3 * u * i * i + i**3 == CURVE_STEPS**3


# --------------------------------------------------------------------------
# conformance


@pytest.mark.skipif(shutil.which("cc") is None or shutil.which("make") is None,
                    reason="no C toolchain")
def test_port_matches_reference_on_a_small_corpus():
    proc = subprocess.run(
        ["python3", str(ROOT / "scripts" / "conformance.py"), "--quick", "--programs", "60"],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "traces identical" in proc.stdout


@pytest.mark.skipif(shutil.which("qemu-system-arm") is None, reason="no qemu-system-arm")
def test_thumb_build_matches_reference_under_qemu():
    """The only path where the port's own instructions execute.

    Everything else in this file measures or reasons about native code; this
    runs the Cortex-M0 image. Skipped rather than failed without QEMU, because
    a missing emulator is not a defect in the port.
    """
    proc = subprocess.run(
        ["python3", str(ROOT / "scripts" / "conformance.py"), "--qemu", "--quick",
         "--programs", "40"],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "traces identical" in proc.stdout
