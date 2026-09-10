"""Sampler settings are data: validate them, key them, and mask control tokens."""

import argparse

import pytest
import torch

from dm.eval.metrics import sample_programs
from dm.eval.sampling import SamplingConfig, parse_temperature, parse_top_k
from dm.isa.codec import BOS, PAD, ByteCodec
from dm.isa.spec import Op


def test_sampler_slug_names_both_controls_and_canonicalises_numbers():
    assert SamplingConfig(40, 1).slug == SamplingConfig(40, 1.0).slug == "k40_t1p0_legal"
    assert SamplingConfig(None, 1.2).slug == "kall_t1p2_legal"
    assert SamplingConfig(20, 0.8).as_dict() == {
        "top_k": 20, "temperature": 0.8, "forbid_specials": True,
        "mask_mode": None, "policy_digest": None,
    }


def test_a_legality_mask_extends_the_sampler_identity_without_moving_old_keys():
    """The state branch adds `mask_mode` and `policy_digest`
    (`docs/state.md`). A mask is part of a generation result's
    specification, so it has to be *in the key* -- and every key written before it
    existed has to stay byte-identical, or two reports of one setting stop being
    comparable for a reason that is not the setting."""
    assert SamplingConfig(40, 1.0).slug == "k40_t1p0_legal"
    masked = SamplingConfig(40, 1.0, "canonical", "cf20e79b115e")
    assert masked.slug == "k40_t1p0_legal_canonical_cf20e79b115e"
    assert masked.as_dict()["mask_mode"] == "canonical"
    # `None` is not `raw`: one says the question was never asked, the other says a
    # mask was available and deliberately not applied.
    assert SamplingConfig(40, 1.0, "raw", "cf20e79b115e").slug != masked.slug
    assert SamplingConfig(40, 1.0).as_dict()["mask_mode"] is None


def test_legality_sampler_identity_refuses_unkeyed_or_unknown_masks():
    for args in (
        (40, 1.0, "canonical", None),
        (40, 1.0, "typo", "cf20e79b115e"),
        (40, 1.0, None, "cf20e79b115e"),
        (40, 1.0, "raw", "not-a-digest"),
    ):
        with pytest.raises(ValueError):
            SamplingConfig(*args)


def test_sampler_arguments_refuse_values_that_collapse_or_invert_probabilities():
    assert parse_top_k("none") is None
    assert parse_top_k("all") is None
    assert parse_top_k("80") == 80
    assert parse_temperature("1.2") == 1.2
    for value in ("0", "-1", "wat"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_top_k(value)
    for value in ("0", "-0.1", "nan", "inf", "wat"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_temperature(value)


def test_raw_and_legal_support_are_distinct_report_identities():
    assert SamplingConfig().slug.endswith("_legal")
    assert SamplingConfig().as_dict()["forbid_specials"] is True


class StubModel:
    def __init__(self) -> None:
        self.forbid = None

    def generate(self, n, max_new, **kwargs):
        self.forbid = kwargs["forbid"]
        # Byte token 2 is bytecode 0x00 (HALT); trailing PAD is dropped.
        monitor = kwargs["monitor"]
        tokens = torch.tensor([[2, PAD]] * n)
        monitor.step(tokens[:, :1].t().numpy())
        return tokens


def test_raw_support_remains_default_for_training_compatibility():
    model = StubModel()
    sample_programs(model, ByteCodec(), n=1, max_new=2)
    assert model.forbid == ()


def test_program_sampling_forbids_model_control_symbols_when_requested():
    model = StubModel()
    programs, _ = sample_programs(
        model, ByteCodec(), n=3, max_new=2, forbid_specials=True,
    )

    assert model.forbid == (PAD, BOS)
    assert programs == [b"\x00"] * 3


def test_cap_hit_comes_from_monitor_state_not_returned_early_stop_width():
    class Staggered:
        def generate(self, n, max_new, **kwargs):
            monitor = kwargs["monitor"]
            # Row 0 halts immediately. Row 1 emits MOVE + two operands, then
            # HALT. Returned width equals row 1's valid length, which used to
            # make row 1 look capped even though both monitor rows are done.
            tokens = torch.tensor([
                [int(Op.HALT) + 2, PAD, PAD, PAD],
                [int(Op.MOVE) + 2, 10 + 2, 20 + 2, int(Op.HALT) + 2],
            ])
            for i in range(tokens.shape[1]):
                monitor.step(tokens[:, i : i + 1].t().numpy())
            return tokens

    programs, cap_hit = sample_programs(Staggered(), ByteCodec(), n=2, max_new=8)

    assert programs == [bytes([int(Op.HALT)]),
                        bytes([int(Op.MOVE), 10, 20, int(Op.HALT)])]
    assert cap_hit == [False, False]


def test_a_live_monitor_row_is_a_real_cap_hit():
    class NeverHalts:
        def generate(self, n, max_new, **kwargs):
            monitor = kwargs["monitor"]
            tokens = torch.tensor([[int(Op.MOVE) + 2, 12, 22]])
            for i in range(tokens.shape[1]):
                monitor.step(tokens[:, i : i + 1].t().numpy())
            return tokens

    _, cap_hit = sample_programs(NeverHalts(), ByteCodec(), n=1, max_new=3)

    assert cap_hit == [True]
