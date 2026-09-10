"""Shared engineering fixture and recipe; no benchmark Adapter dependencies."""
from dataclasses import replace

from dm.data import relation
from dm.isa.codec import ByteCodec
from dm.train_relation import RelationTrainConfig, TrainingCorpus

MODEL_CONFIG = {
    "vocab_size": ByteCodec.vocab_size,
    "d_model": 16,
    "n_layers": 1,
    "n_heads": 2,
    "max_len": 256,
}


def fixture() -> TrainingCorpus:
    built = relation.build_venue1(
        8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples()
    )
    cases = list(built.cases)
    # Annotation-negative only: original bytes can still encode a relation.
    cases[1] = replace(cases[1], actions=())
    return TrainingCorpus.engineering_cases(cases)


def training_config(arm: str, budget: int, *, steps: int, warmup: int) -> RelationTrainConfig:
    return RelationTrainConfig(
        arm,
        {**MODEL_CONFIG, "relation_schema": arm},
        seed=31,
        batch_size=3,
        max_len=256,
        steps=steps,
        warmup=warmup,
        attention_budget=budget,
        corpus_provenance="engineering",
        seed_namespace="engineering",
    )
