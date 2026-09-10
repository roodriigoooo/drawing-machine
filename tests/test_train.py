"""The batching machinery exists to cut memory and padding, not to change the
experiment. These tests pin the two properties that guarantee that: splitting a
batch leaves the gradient identical, and bucketing still shows every program
exactly once per epoch."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import dm.train
from dm.data.dataset import PAD, BucketedBatchSampler, ProgramDataset, loader
from dm.data.synthetic import dataset as synth
from dm.isa.codec import CODECS
from dm.models.transformer import Config, DrawingLM
from dm.train import TrainConfig, micro_batches, train


def _grad_of(model, inputs, targets, budget):
    n_tokens = (targets != PAD).sum().clamp(min=1)
    model.zero_grad(set_to_none=True)
    for chunk_in, chunk_tgt, _ in micro_batches(inputs, targets, budget):
        logits = model(chunk_in)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), chunk_tgt.reshape(-1),
            ignore_index=PAD, reduction="sum",
        )
        (nll / n_tokens).backward()
    return [p.grad.clone() for p in model.parameters()]


def test_accumulation_does_not_change_the_gradient():
    torch.manual_seed(0)
    cfg = Config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, max_len=64)
    model = DrawingLM(cfg)
    inputs = torch.randint(2, 32, (8, 16))
    targets = inputs.clone()
    targets[:, 12:] = PAD  # uneven padding: the normaliser has to be token-based

    split_budget = 2 * 16 * 16  # rows x T^2, so two rows per chunk
    whole = _grad_of(model, inputs, targets, budget=10**9)
    split = _grad_of(model, inputs, targets, budget=split_budget)
    assert len(list(micro_batches(inputs, targets, split_budget))) == 4, "budget did not split"
    for a, b in zip(whole, split):
        assert torch.allclose(a, b, atol=1e-6)


def test_micro_batches_never_yield_an_empty_chunk():
    inputs = torch.zeros(4, 4096, dtype=torch.long)
    chunks = list(micro_batches(inputs, inputs, budget=1))
    assert len(chunks) == 4 and all(c[0].shape[0] == 1 for c in chunks)


def test_micro_batches_slice_the_conditioning_with_its_own_rows():
    """The one way a conditional run can be wrong while looking perfectly
    healthy: a class tensor sliced against a different offset than the rows it
    belongs to trains every drawing on a neighbour's category, and the loss curve
    is indistinguishable from a correct one."""
    inputs = torch.arange(24, dtype=torch.long).reshape(6, 4)
    classes = torch.arange(6)
    chunks = list(micro_batches(inputs, inputs, budget=2 * 4 * 4, classes=classes))
    assert len(chunks) == 3
    for chunk_in, _, chunk_cls in chunks:
        assert chunk_cls is not None and len(chunk_cls) == len(chunk_in)
        # Row r carries class r, whichever chunk it landed in.
        assert torch.equal(chunk_cls, chunk_in[:, 0] // 4)
    assert all(c[2] is None for c in micro_batches(inputs, inputs, budget=10**9))


def test_bucketing_covers_every_program_once():
    lengths = [7, 3, 99, 3, 41, 5, 60, 12, 12, 1]
    for shuffle in (False, True):
        sampler = BucketedBatchSampler(lengths, batch_size=3, shuffle=shuffle, pool_batches=2)
        batches = list(sampler)
        assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
        assert len(batches) == len(sampler)


def test_sampler_epoch_order_uses_its_own_epoch_indexed_stream():
    lengths = [7, 3, 99, 3, 41, 5, 60, 12, 12, 1]
    sampler = BucketedBatchSampler(lengths, batch_size=3, pool_batches=2, seed=17)
    control = BucketedBatchSampler(lengths, batch_size=3, pool_batches=2, seed=17)

    first = list(sampler)
    torch.rand(10_000)  # simulates an unrelated generation diagnostic
    second = list(sampler)
    assert first == list(control)
    assert second == list(control)


def test_generation_rng_consumption_cannot_change_cpu_final_weights(tmp_path, monkeypatch):
    """Generation evaluation is allowed to be inserted or removed without
    changing a later epoch's batches or the optimiser trajectory."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    real_sample_quality = dm.train.sample_quality

    baseline = train(_tiny(steps=4, tag="unit_rng_baseline"), verbose=False)
    baseline_weights = torch.load(
        tmp_path / "unit_rng_baseline.pt", weights_only=False
    )["state"]

    def noisy_sample_quality(*args, **kwargs):
        torch.rand(50_000)  # legacy global-stream consumer
        return real_sample_quality(*args, **kwargs)

    monkeypatch.setattr(dm.train, "sample_quality", noisy_sample_quality)
    train(_tiny(steps=4, tag="unit_rng_noisy"), verbose=False)
    noisy_weights = torch.load(
        tmp_path / "unit_rng_noisy.pt", weights_only=False
    )["state"]

    assert baseline["steps_done"] == 4
    assert baseline_weights.keys() == noisy_weights.keys()
    assert all(torch.equal(baseline_weights[name], noisy_weights[name])
               for name in baseline_weights)


def test_bucketing_cuts_padding_waste():
    """The whole point of the sampler. Random batches measured at 2.31x the mean
    length on every codec; attention is quadratic, so that is ~5x the memory."""
    programs = synth(600, seed=0)
    codec = CODECS["bit"]
    dataset = ProgramDataset(programs, codec, 2048)
    stats = dataset.length_stats()
    widths = [x.shape[1] + 1 for _, x, _ in loader(dataset, 64)]
    waste = sum(widths) / len(widths) / stats["mean"]
    assert waste < 1.25, f"bucketed batches still pad {waste:.2f}x past the mean"


def test_loader_indices_identify_the_program_they_scored():
    """Per-program bits are only paired across codecs if the index that comes
    back off the loader is the program's position in the split, not its position
    in whatever order bucketing happened to emit."""
    programs = synth(200, seed=1)
    codec = CODECS["byte"]
    seen = {}
    for index, inputs, targets in loader(ProgramDataset(programs, codec, 2048), 16):
        for i, src, tgt in zip(index.tolist(), inputs, targets):
            # inputs/targets are the sequence offset by one, so the first input
            # plus every target rebuilds it -- which also pins that alignment.
            seen[i] = [int(src[0]), *tgt[tgt != PAD].tolist()]
    assert len(seen) == len(programs)
    for i, program in enumerate(programs):
        assert seen[i] == codec.with_bos(program)


def test_the_record_keeps_the_val_bits_of_its_own_best_checkpoint(tmp_path, monkeypatch):
    """`final` is what gets reported, but `final - best` is the drift, and under
    schema 3 that drift was the size of the effects under test while being
    invisible from the records -- only the final eval's per-program bits were
    kept, so no paired difference could be recomputed at the optimum.

    The invariant is alignment: `best_val_bits` must be the array of the eval
    that `best` names, not whichever one happened to be in the variable last."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    result = train(
        TrainConfig(
            codec="byte", shape="square", n_train=64, n_val=16, steps=3, eval_every=1,
            warmup=1, batch_size=16, max_len=256, gen_samples=4, device="cpu",
            tag="unit_best_checkpoint",
        ),
        verbose=False,
    )

    history = result["history"]
    assert len(history) == 3, "eval_every=1 should leave one record per step"
    assert result["final"] is history[-1]
    assert result["best"]["bits_per_drawing"] == min(h["bits_per_drawing"] for h in history)

    for record, bits in ((result["final"], result["val_bits"]),
                         (result["best"], result["best_val_bits"])):
        assert len(bits) == 16, "one entry per val program, or the pairing breaks"
        # bits_per_drawing is the mean of exactly these numbers, so a lagged or
        # mismatched array shows up here and nowhere else.
        assert abs(sum(bits) / len(bits) - record["bits_per_drawing"]) < 1e-3


# ---------------------------------------------------------------------------
# The record is written at every eval, because a trainer that writes at exit
# writes nothing when it is killed. `docs/history/lost-run.md` is the 2.6 hours
# this cost.


def _tiny(**overrides) -> TrainConfig:
    return TrainConfig(
        **{"codec": "byte", "shape": "square", "n_train": 64, "n_val": 16,
           "steps": 3, "eval_every": 1, "warmup": 1, "batch_size": 16,
           "max_len": 256, "gen_samples": 4, "device": "cpu", **overrides}
    )


def test_engineering_provenance_is_embedded_and_linked(tmp_path, monkeypatch):
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    result = train(
        _tiny(tag="unit_provenance", artifact_provenance="engineering"),
        verbose=False,
    )
    record_path = tmp_path / "unit_provenance.json"
    checkpoint_path = tmp_path / "unit_provenance.pt"
    record = json.loads(record_path.read_text())
    weights = torch.load(checkpoint_path, weights_only=False)
    assert result["provenance"] == record["provenance"] == "engineering"
    assert weights["provenance"] == "engineering"
    assert weights["record_name"] == "unit_provenance"
    assert record["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert record["config"]["artifact_provenance"] == "engineering"


def test_a_killed_run_still_leaves_the_record_it_had_reached(tmp_path, monkeypatch):
    """The property the schema-2 planner re-run needed and did not have.

    It reached step 11,500 of 12,000 over 2.6 hours, took a SIGKILL, and left no
    file -- so the replicate floor it was launched to measure survives only as
    terminal scrollback, without the per-program `val_bits` every claim-3
    difference is quoted with. A kill is simulated here by an exception out of
    the eval, which exercises the same invariant: whatever is on disk when the
    loop stops is a whole record of the evals that did happen.
    """
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    calls = {"n": 0}
    real = dm.train.bits_per_drawing

    def die_on_the_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt("pretend this was a SIGKILL")
        return real(*args, **kwargs)

    monkeypatch.setattr(dm.train, "bits_per_drawing", die_on_the_third)
    with pytest.raises(KeyboardInterrupt):
        train(_tiny(tag="unit_killed"), verbose=False)

    record = json.loads((tmp_path / "unit_killed.json").read_text())
    assert record["complete"] is False, "a run that did not return must not say it did"
    assert record["steps_done"] == 2, "the record names the last eval that landed"
    assert record["config"]["steps"] == 3, "and keeps the budget it was asked for"
    assert len(record["history"]) == 2
    assert len(record["val_bits"]) == 16, "the paired array survives, or the record cannot be differenced"
    assert (tmp_path / "unit_killed.pt").exists(), "and the weights are on disk too"


def test_the_record_says_whether_the_trainer_returned(tmp_path, monkeypatch):
    """`complete` means "train() returned", not "the loop reached its last step".

    The two differ exactly when a kill lands between the final eval and the
    return, and a record that claimed the stronger thing there would be wrong in
    the one case the flag exists to catch.
    """
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    result = train(_tiny(tag="unit_complete"), verbose=False)
    on_disk = json.loads((tmp_path / "unit_complete.json").read_text())

    assert result["complete"] is True and on_disk["complete"] is True
    assert result["steps_done"] == on_disk["steps_done"] == 3
    assert on_disk["final"]["step"] == 3
    # The returned dict and the published file are the same record. They were
    # written by the same call, and a reader that trusts one has to be able to
    # trust the other.
    assert on_disk["history"] == result["history"]


def test_the_record_is_published_atomically_and_after_its_weights(tmp_path, monkeypatch):
    """Both halves of `dm.train.checkpoint`'s contract, pinned at the mechanism.

    *Atomically*, because `scripts/sweep.py` globs `runs/*.json` while a run is
    training and every reader assumes a whole record: a kill inside a plain
    `write_text` leaves half of one at the path where a whole one is expected.
    *Weights first*, because the record is the commit marker -- a kill between
    the two writes then leaves weights that no record claims, which costs
    nothing, rather than a record whose `.pt` holds a different model.
    """
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    published: list[tuple[Path, Path]] = []
    real = os.replace

    def watch(src, dst):
        published.append((Path(src), Path(dst)))
        return real(src, dst)

    monkeypatch.setattr(dm.train.os, "replace", watch)
    train(_tiny(steps=1, tag="unit_atomic"), verbose=False)

    assert published, "nothing went through os.replace, so nothing was staged"
    for src, dst in published:
        assert src != dst, "written straight to the destination, so a kill truncates it"
        assert src.parent == dst.parent, (
            "staged on another filesystem, where os.replace is not atomic"
        )
    # One eval plus the completing rewrite, weights before record each time.
    assert [dst.suffix for _, dst in published] == [".pt", ".json", ".pt", ".json"]
    assert not list(tmp_path.glob(".*tmp")), "a staging file outlived its write"
