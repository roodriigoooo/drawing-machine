"""Model-blind, fully expanded experiment recipe for the bounded affine task."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

from dm.isa.codec import ByteCodec
from dm.models.transformer import Config
from dm.relation.demonstrated import NAMESPACE
from dm.relation.demonstrated_fixture import build_fit_fixture
from dm.relation.demonstrated_pilot import build_pilot_fixture
from dm.train_relation import RelationBucketCursor, RelationTrainConfig, TrainingCorpus

CELLS = (("fit-131", "fit", "span_affine_v1", 131),
         ("fit-137", "fit", "span_affine_v1", 137),
         ("pilot-a-none", "pilot", "none", 775159550),
         ("pilot-a-relation", "pilot", "span_affine_v1", 775159550),
         ("pilot-b-relation", "pilot", "span_affine_v1", 728883118),
         ("pilot-b-none", "pilot", "none", 728883118))


def config(arm, seed):
    model = Config(vocab_size=ByteCodec.vocab_size, d_model=128, n_layers=4,
                   n_heads=4, max_len=256, relation_schema=arm)
    if model.n_params() >= 1_000_000:
        raise ValueError("intended model exceeds parameter cap")
    return RelationTrainConfig(arm, asdict(model), seed, 8, 256, 1024, 64,
                               corpus_provenance="engineering", seed_namespace="engineering")


def uniforms(split, case_id):
    return [int.from_bytes(hashlib.sha256(
        f"{NAMESPACE}:{split}:{case_id}:{position}".encode()).digest()[:3], "big") / 2**24
        for position in range(256)]


def recipe():
    fit = build_fit_fixture().cases
    panels = dict(build_pilot_fixture().panels)
    cells = []
    for name, stage, arm, seed in CELLS:
        cases = fit if stage == "fit" else panels["train"]
        training = TrainingCorpus.engineering_cases(cases)
        cfg = config(arm, seed)
        cursor = RelationBucketCursor.seeded([len(c.flat) + 1 for c in cases], 8, seed)
        cells.append({"cell_id": name, "stage": stage, "config": asdict(cfg),
                      "parameter_count": Config(**cfg.model).n_params(),
                      "program_fingerprint": training.program_fingerprint,
                      "supervision_identity": training.supervision_identity,
                      "batch_order": [cursor.next() for _ in range(1024)],
                      "evaluation_steps": [0, 64, 256, 1024] if stage == "fit" else [0, 128, 512, 1024]})
    # The pilot readout is predeclared and balanced, with both negative classes.
    intermediate = panels["train"][:16] + panels["train"][512:520] + panels["train"][768:776]
    populations = {"fit": fit, **panels}
    return {"namespace": NAMESPACE, "cells": cells,
            "pilot_intermediate_case_ids": [c.case_id for c in intermediate],
            "uniforms": {split: {c.case_id: uniforms(split, c.case_id) for c in cases}
                         for split, cases in populations.items()},
            "decode": {"literal_policy": "content_bytes_v1", "stopping_policy": "flat_l0_stop_v1",
                       "max_output_bytes": 256, "temperature": 1.0,
                       "uniform_recipe": "sha256(namespace:split:case_id:absolute_byte_position) first24bits / 2**24",
                       "pairing": "same position-keyed float32 stream across arms and modes within seed",
                       "target_length_is_decoder_input": False},
            "budgets": {"maximum_optimizer_updates": 6144, "aggregate_seconds": 14400,
                        "per_cell_seconds_including_evaluation": 5400},
            "progression": {"fit_both_seeds_positive_exact_copy_minimum": 15,
                            "fit_positive_denominator": 16, "fit_negative_copy_maximum": 0,
                            "fit_negative_denominator": 16,
                            "pilot_requires_both_fit_seeds_pass": True,
                            "no_automatic_extension_or_tuning": True},
            "pilot": {"bootstrap_draws": 2000, "bootstrap_seed": 20260914,
                      "bootstrap_generator": "numpy.default_rng/PCG64; percentile linear quantiles",
                      "conditional_correct_minimum": 60,
                      "training_panel_fit": "same 15/16 exact COPY and zero negative invocation rule as I1",
                      "selection_bottleneck": "both seeds: all 64 affine conditional actions correct with negative margins",
                      "interpretation_precedence": ["unsafe_copying", "weak_training_or_standard_regression",
                                                    "replicated_useful_pilot_signal", "selection_bottleneck",
                                                    "fit_without_transfer", "transfer_without_generation_benefit",
                                                    "mixed_or_inconclusive"],
                      "validity_screen_population": "all 320 final held-out programs, not training readouts",
                      "bootstrap_unit": "connected_source_and_donor_component",
                      "affine_D_copy_minimum": 0.10, "gap_recovery_minimum": 0.50,
                      "descriptive_interval_lower_bound_exclusive": 0,
                      "negative_invocations_maximum": 1, "negative_denominator": 128,
                      "valid_halt_loss_maximum": 0.01,
                      "standard_extra_bits_per_program_maximum": 1,
                      "standard_valid_halt_loss_vs_none_maximum": 0.05}}
