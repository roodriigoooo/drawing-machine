"""Programs -> padded token batches, one codec at a time.

Sequences are left-aligned with PAD on the right and the loss ignores PAD, so
padded positions cannot influence any real position under causal attention.

Batches are assembled from similar-length programs. Uniformly random batches
pad to near the corpus maximum -- measured at 2.31x the mean length on all four
codecs -- and attention is quadratic, so that factor costs the bit arm ~5x its
memory for positions the loss then discards.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..isa.codec import BOS, PAD, Codec


class ProgramDataset(Dataset):
    """Encoded programs, and optionally the class each one belongs to.

    **`labels` rides on the dataset rather than through `collate`, and that is
    the whole conditioning convention on the scoring side.** Every batch already
    carries its `index` -- it has to, because bucketing reorders programs and a
    paired comparison needs the score attributed back. So a scorer that wants the
    conditioning signal looks up `dataset.labels[index]`, and `collate`, `loader`
    and every existing caller stay exactly as they were. Sampling has no dataset
    and takes its classes explicitly (`dm.eval.metrics.sample_programs`).
    """

    def __init__(self, programs: list[bytes], codec: Codec, max_len: int = 2048,
                 labels: list[int] | None = None) -> None:
        self.codec, self.max_len = codec, max_len
        if labels is not None and len(labels) != len(programs):
            raise ValueError(
                f"{len(labels)} labels for {len(programs)} programs; a misaligned "
                "label array conditions every drawing on the wrong class and "
                "nothing downstream can notice"
            )
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)
        # Truncation is counted, not assumed away. `max_len` cuts the *codec's*
        # symbols, and the bit codec emits 8x as many for the same drawing, so a
        # corpus that fits comfortably for byte can be cut for bit alone -- and a
        # cut program is scored on fewer bits than it costs, which biases the one
        # metric that is comparable across codecs, on exactly the granularity
        # axis under test. Tier A never reached `max_len`; Tier B does, so this
        # has to be visible in the run record rather than discovered later.
        self.seqs, self.n_truncated = [], 0
        for program in programs:
            encoded = codec.with_bos(program)
            self.n_truncated += len(encoded) > max_len
            self.seqs.append(torch.tensor(encoded[:max_len], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        return i, self.seqs[i]

    def length_stats(self) -> dict[str, float]:
        lens = sorted(len(s) for s in self.seqs)
        return {
            "mean": sum(lens) / len(lens),
            "p50": float(lens[len(lens) // 2]),
            "p99": float(lens[int(0.99 * (len(lens) - 1))]),
            "max": float(lens[-1]),
            "truncated": self.n_truncated / max(1, len(lens)),
        }


def collate(batch: list[tuple[int, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    """Rows carry their dataset index.

    Bucketing reorders programs, so without the index a per-program score cannot
    be attributed back to a program -- which is what a paired comparison across
    codecs needs, given the effects under test are ~1 bit against a ~3.5 bit
    spread between val sets.
    """
    width = max(len(s) for _, s in batch)
    padded = torch.full((len(batch), width), PAD, dtype=torch.long)
    for i, (_, seq) in enumerate(batch):
        padded[i, : len(seq)] = seq
    index = torch.tensor([i for i, _ in batch], dtype=torch.long)
    return index, padded[:, :-1], padded[:, 1:]


class BucketedBatchSampler(Sampler[list[int]]):
    """Batches of similar-length sequences, still shuffled.

    Sorting the whole corpus once would fix batch composition for the entire
    run, so lengths are sorted only inside a shuffled pool of `pool_batches`
    batches. Composition is fresh every epoch and padding is to the local
    maximum rather than the global one, which is where the 2.31x goes.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True,
                 pool_batches: int = 50, seed: int = 0) -> None:
        self.lengths, self.batch_size = lengths, batch_size
        self.shuffle, self.pool_batches = shuffle, pool_batches
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        # The sampler is a separate random stream.  In particular, do not use
        # torch's global generator here: training-time generation diagnostics
        # also draw from that stream, and a model output must never choose the
        # order of the next epoch.  Deriving the seed from the epoch makes every
        # epoch independently reproducible while retaining fresh shuffles.
        epoch = self._epoch
        self._epoch += 1
        generator = torch.Generator().manual_seed(
            (self.seed + 1_000_003 * epoch) % (2**63 - 1)
        )
        order = (torch.randperm(n, generator=generator).tolist()
                 if self.shuffle else list(range(n)))
        pool = self.batch_size * self.pool_batches
        batches = [
            chunk[i : i + self.batch_size]
            for start in range(0, n, pool)
            for chunk in [sorted(order[start : start + pool], key=self.lengths.__getitem__)]
            for i in range(0, len(chunk), self.batch_size)
        ]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(
                len(batches), generator=generator).tolist()]
        return iter(batches)


def loader(dataset: ProgramDataset, batch_size: int = 64, shuffle: bool = True,
           seed: int = 0) -> DataLoader:
    """Wraps an already-encoded dataset, so the caller owns the encoding.

    Taking programs here instead invites a second `ProgramDataset` over the same
    corpus whenever the caller also wants its length statistics -- a full
    re-encode, which at 100k programs on the bit arm is ~2s and ~0.3 GB per run.
    """
    return DataLoader(
        dataset,
        batch_sampler=BucketedBatchSampler(
            [len(s) for s in dataset.seqs], batch_size, shuffle=shuffle,
            seed=seed,
        ),
        collate_fn=collate,
        # DataLoader itself may draw an iterator base seed.  Give it a distinct
        # private generator too, so that this implementation detail cannot touch
        # the model/evaluation stream even when PyTorch changes its iterator
        # bookkeeping.
        generator=torch.Generator().manual_seed(
            (int(seed) + 2_000_003) % (2**63 - 1)
        ),
    )


__all__ = ["BOS", "PAD", "BucketedBatchSampler", "ProgramDataset", "collate", "loader"]
