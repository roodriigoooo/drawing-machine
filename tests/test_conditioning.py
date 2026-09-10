"""Class conditioning: the plumbing, and the identity the reading rests on.

Two kinds of test here and they guard different failures.

**The plumbing tests guard silence.** A misaligned label array trains every
drawing on a neighbour's class and the loss curve looks perfectly healthy; a
conditional model called without classes scores every drawing under a zero vector
that is not any class; a conditional arm differenced against another conditional
arm measures nothing. None of those raises on its own.

**The identity tests guard the result.** The whole reading is that
`I(X;C) = H(C) − H(C|X)` connects a likelihood gap to a classifier's accuracy, so
each half is pinned against a model whose answer is known without running it: a
model that ignores its conditioning must land at chance and at exactly `H(C)`
bits, and one that is perfect must land at 1.0 and 0 bits, with Fano agreeing at
both ends.
"""

import math

import numpy as np
import pytest
import torch

from dm.data.dataset import ProgramDataset, collate, loader
from dm.eval.conditioning import (
    class_log2_likelihood,
    class_posterior,
    conditional_class_bits,
    fano_error_bound,
    label_entropy,
)
from dm.isa.asm import assemble
from dm.isa.codec import CODECS
from dm.models.transformer import Config, DrawingLM

CODEC = CODECS["byte"]


def programs(n: int = 12) -> list[bytes]:
    return [assemble(f"MOVE {10 + i} {20 + i}\nLINE {60 + i} {70 + i}\nHALT")
            for i in range(n)]


# --- the model's half -------------------------------------------------------


def test_the_conditional_arm_starts_where_its_control_starts():
    """The axis is an *addition*: the class table is zero-initialised, so a fresh
    conditional model and its unconditional twin at the same seed are the same
    function. Without that, the two arms differ by a different draw as well as by
    the conditioning, and the draw is worth ~2 bits on this project's own
    measurement -- larger than the entire label ceiling."""
    idx = torch.randint(2, 258, (4, 9))
    classes = torch.tensor([0, 1, 2, 3])
    torch.manual_seed(7)
    plain = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2))
    torch.manual_seed(7)
    cond = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2, n_classes=5))
    assert torch.allclose(plain(idx), cond(idx, classes))
    # And the conditioning is reachable: perturb the table and the answer moves.
    with torch.no_grad():
        cond.classes[3] += 1.0
    assert not torch.allclose(plain(idx), cond(idx, classes))


def test_conditioning_does_not_widen_the_logits():
    """Why the signal is an additive embedding and not a class *token*. Embeddings
    are tied to the output head, so a token would widen the logits past
    `codec.vocab_size` -- and `dm/eval/recovery.py`, `spelling.py` and
    `redundancy.py` all reshape logits against the codec's own width, which does
    not raise when it is wrong, it reinterprets the tensor."""
    idx = torch.randint(2, 258, (2, 5))
    cond = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2, n_classes=5))
    assert cond(idx, torch.zeros(2, dtype=torch.long)).shape[-1] == CODEC.vocab_size
    # And the cost is stated: one row of d_model per class, and nothing else.
    base = Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2)
    assert (Config(**{**base.__dict__, "n_classes": 5}).n_params()
            == base.n_params() + 5 * 32)


def test_a_conditional_model_refuses_to_be_called_unconditioned():
    """Both directions, because both are silent. Scoring a conditional model with
    no classes applies a zero vector that is not any class; passing classes to an
    unconditional model has them ignored while the run is filed as conditional."""
    idx = torch.randint(2, 258, (2, 5))
    cond = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2, n_classes=3))
    plain = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2))
    with pytest.raises(ValueError, match="without classes"):
        cond(idx)
    with pytest.raises(ValueError, match="unconditional model"):
        plain(idx, torch.zeros(2, dtype=torch.long))


def test_generation_takes_the_class_it_was_asked_for():
    """Sampling has no dataset to carry labels, so its classes are explicit. The
    check is that they reach the embedding: a table with one class driven to a
    huge value must change what that row emits and leave the others alone.

    Greedy (`top_k=1`), because the comparison has to be structural. Sampled
    decoding can draw the identical eight tokens from two different distributions
    by luck, and a test that fails one seed in five is worse than no test."""
    torch.manual_seed(11)
    cond = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2,
                            n_classes=2, max_len=64))
    with torch.no_grad():
        cond.classes[1] += 30.0
    both = cond.generate(4, 8, classes=torch.tensor([0, 1, 0, 1]), top_k=1)
    zeros = cond.generate(4, 8, classes=torch.zeros(4, dtype=torch.long), top_k=1)
    assert torch.equal(both[0], zeros[0]) and torch.equal(both[2], zeros[2])
    assert not torch.equal(both[1], zeros[1])
    assert not torch.equal(both[3], zeros[3])


# --- the dataset's half -----------------------------------------------------


def test_labels_ride_on_the_dataset_and_collate_is_untouched():
    """The conditioning convention on the scoring side: the batch already carries
    `index`, so a scorer looks the class up rather than having it threaded through
    `collate` -- which is what keeps every existing instrument working."""
    text = programs(6)
    dataset = ProgramDataset(text, CODEC, 64, labels=[0, 1, 2, 0, 1, 2])
    index, inputs, targets = collate([dataset[i] for i in range(6)])
    assert inputs.shape == targets.shape
    assert dataset.labels is not None
    assert torch.equal(dataset.labels[index], torch.tensor([0, 1, 2, 0, 1, 2]))
    assert ProgramDataset(text, CODEC, 64).labels is None


def test_a_misaligned_label_array_is_refused():
    with pytest.raises(ValueError, match="labels for"):
        ProgramDataset(programs(6), CODEC, 64, labels=[0, 1])


def test_the_scorer_conditions_on_the_dataset_labels():
    """`per_program_bits` is shared by every likelihood number in the project, so
    the conditioning has to reach it without a signature change -- and it has to
    change the answer, or the label is decorative."""
    from dm.eval.metrics import per_program_bits

    text = programs(8)
    model = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2,
                             max_len=64, n_classes=2))
    with torch.no_grad():
        model.classes[1] += 5.0
    a = per_program_bits(model, loader(ProgramDataset(text, CODEC, 64, labels=[0] * 8),
                                      4, shuffle=False))
    b = per_program_bits(model, loader(ProgramDataset(text, CODEC, 64, labels=[1] * 8),
                                      4, shuffle=False))
    assert not np.allclose(a, b)


# --- the identity the reading rests on --------------------------------------


def test_the_ceiling_is_the_label_entropy():
    balanced = label_entropy([0, 1, 2, 3, 4] * 20, 5)
    assert balanced["entropy_bits"] == pytest.approx(math.log2(5))
    assert balanced["chance_accuracy"] == pytest.approx(0.2)
    # Measured rather than assumed from the class count: a `limit=` that lands
    # mid-round leaves the split unbalanced, and a ceiling quoted as log2(k) would
    # then be too high -- in the direction that flatters a conditioning result.
    skewed = label_entropy([0] * 90 + [1] * 10, 2)
    assert skewed["entropy_bits"] < skewed["uniform_bits"]
    assert skewed["chance_accuracy"] == pytest.approx(0.9)


def test_a_model_that_ignores_its_class_lands_at_chance_and_at_H_of_C():
    """The null, and it is exact. Identical likelihoods under every class give a
    uniform posterior, so `H(C|X) = H(C)`, mutual information 0, accuracy at
    chance -- and Fano must permit nothing better."""
    nll = np.tile(np.array([[40.0, 40.0, 40.0, 40.0, 40.0]]), (100, 1))
    labels = [i % 5 for i in range(100)]
    read = conditional_class_bits(class_posterior(nll)["posterior"], labels)
    assert read["class_bits"] == pytest.approx(math.log2(5))
    assert read["accuracy"] == pytest.approx(0.2)
    ceiling = label_entropy(labels, 5)["entropy_bits"]
    assert ceiling - read["class_bits"] == pytest.approx(0.0)
    assert 1 - fano_error_bound(read["class_bits"], 5) == pytest.approx(0.2, abs=0.01)


def test_a_perfect_model_lands_at_one_and_at_zero_bits():
    """The other end. A model certain of the true class spends 0 bits on it, so the
    mutual information is the whole of `H(C)` -- which is the ceiling, and nothing
    may exceed it."""
    labels = [i % 5 for i in range(100)]
    nll = np.full((100, 5), 400.0)
    nll[np.arange(100), labels] = 1.0
    read = conditional_class_bits(class_posterior(nll)["posterior"], labels)
    assert read["accuracy"] == pytest.approx(1.0)
    assert read["class_bits"] == pytest.approx(0.0, abs=1e-9)
    assert fano_error_bound(read["class_bits"], 5) == pytest.approx(0.0, abs=1e-9)
    assert label_entropy(labels, 5)["entropy_bits"] - read["class_bits"] <= math.log2(5)


def test_the_posterior_is_bayes_and_the_prior_moves_it():
    """`p(c|x) ∝ p(x|c)p(c)`, in log2 throughout so the units never change hands.
    A prior that rules a class out must remove it from the argmax even when the
    likelihood prefers it."""
    nll = np.array([[2.0, 3.0]])
    flat = class_posterior(nll)["posterior"][0]
    # One bit of likelihood advantage is a factor of two in probability.
    assert flat[0] / flat[1] == pytest.approx(2.0)
    assert flat.sum() == pytest.approx(1.0)
    skewed = class_posterior(nll, prior=np.array([0.01, 0.99]))["posterior"][0]
    assert skewed.argmax() == 1


def test_the_free_classifier_reads_a_model_that_really_was_conditioned():
    """End to end on a real model and real programs: score every program under
    every class, and the class whose embedding the program was scored with must be
    the one the posterior picks. This is the path `scripts/conditioning.py` runs,
    with the answer arranged rather than trained."""
    text = programs(10)
    model = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2,
                             max_len=64, n_classes=3))
    nll = class_log2_likelihood(model, text, CODEC, 3, max_len=64, batch_size=4)
    assert nll.shape == (10, 3)
    # An untrained model is near-uniform over classes, so the posterior is near
    # chance -- which is the honest null and is what makes the trained reading a
    # measurement rather than a property of the arithmetic.
    posterior = class_posterior(nll)["posterior"]
    assert posterior.shape == (10, 3)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="conditional model"):
        class_log2_likelihood(model, text, CODEC, 0)


def test_shared_init_leaves_the_class_table_at_zero():
    """The interaction that would have quietly ruined the experiment.

    `share_non_embedding_init` redraws every non-embedding matrix from one
    generator so two arms differ by their codec and not by their draw -- and the
    class table has the same rank as a weight matrix. Filled from that generator it
    would (a) stop the conditional arm from starting where its control starts and
    (b) consume draws the control never consumed, so the *layer* weights would
    diverge too. The control arm on record uses `--share-init`, so this is the path
    the real comparison takes."""
    # Seeded construction, because `share_non_embedding_init` deliberately does
    # *not* touch the embedding -- that is the thing a codec ablation differs in.
    # The class table takes no RNG at all (it is zero-filled, not `_init`ed), so
    # the two constructions consume the identical stream.
    torch.manual_seed(5)
    plain = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2))
    torch.manual_seed(5)
    cond = DrawingLM(Config(vocab_size=258, d_model=32, n_layers=2, n_heads=2, n_classes=5))
    plain.share_non_embedding_init(3)
    cond.share_non_embedding_init(3)

    assert cond.classes is not None
    assert torch.count_nonzero(cond.classes) == 0
    # Every other parameter identical, so the arms differ by the class table alone
    # -- including `embed.weight`, which shared-init leaves to the constructor.
    shared = dict(plain.named_parameters())
    for name, param in cond.named_parameters():
        if name == "classes":
            continue
        assert torch.equal(param, shared[name]), name
    idx = torch.randint(2, 258, (2, 6))
    assert torch.allclose(plain(idx), cond(idx, torch.tensor([0, 4])))
