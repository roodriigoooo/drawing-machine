"""Score two causal attention windows on the frozen QuickDraw demo and its validation set.

Run from the repository root with PYTHONPATH=. and the training dependencies.
Uses the local checkpoint and converted corpus. Outputs a descriptive JSON record.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dm.data.dataset import ProgramDataset, collate
from dm.data.fingerprint import digest
from dm.data.quickdraw import load_labelled
from dm.demo.sample import load_checkpoint
from dm.eval.metrics import per_program_bits

CHECKPOINT = Path("runs/quickdraw_cond24000eps4_byte_square_s0.pt")


def score_window(model, loader, width):
    """Mask the attention scores in every layer while preserving RoPE positions."""
    full_attention = F.scaled_dot_product_attention

    def attend(query, key, value, *, is_causal, dropout_p):
        assert is_causal and dropout_p == 0 and query.size(-2) == key.size(-2)
        position = torch.arange(key.size(-2), device=key.device)
        distance = position[:, None] - position[None, :]
        allowed = distance >= 0
        if width is not None:
            allowed &= distance < width
        return full_attention(query, key, value, attn_mask=allowed)

    with patch.object(F, "scaled_dot_product_attention", attend):
        return per_program_bits(model, loader)


def main():
    checkpoint = load_checkpoint(CHECKPOINT)
    record = checkpoint.record
    programs, classes = load_labelled(checkpoint.classes, "valid", rdp_eps=4.0)
    count = record["corpus"]["n_val"]
    programs = programs[:count]
    lengths = np.array([len(p) for p in programs])
    assert digest(programs) == record["corpus"]["val"]
    assert lengths.tolist() == record["val_bytes"]
    assert classes is not None
    dataset = ProgramDataset(programs, checkpoint.codec, record["model"]["max_len"], classes[:count])
    assert dataset.n_truncated == 0
    order = np.argsort(lengths, kind="stable").tolist()
    batches = [order[i : i + 16] for i in range(0, count, 16)]
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate)

    torch.set_num_threads(4)
    ordinary = per_program_bits(checkpoint.model, loader)
    full = score_window(checkpoint.model, loader, None)
    max_mask_error = float(np.abs(full - ordinary).max())
    assert max_mask_error < 0.002, "an explicit full causal mask changed the reference scores"

    windows = []
    for width in (128, 192):
        difference = score_window(checkpoint.model, loader, width) - full
        affected = lengths > width
        assert np.abs(difference[~affected]).max() < 0.002
        tail = int((lengths[affected] - width).sum())
        windows.append({
            "window_positions": width,
            "affected_drawings": int(affected.sum()),
            "affected_target_bytes": tail,
            "mean_extra_bits_per_drawing": float(difference.mean()),
            "mean_extra_bits_per_affected_drawing": float(difference[affected].mean()),
            "median_extra_bits_per_affected_drawing": float(np.median(difference[affected])),
            "extra_bits_per_affected_target_byte": float(difference[affected].sum() / tail),
            "affected_min_max_extra_bits": [float(difference[affected].min()),
                                            float(difference[affected].max())],
        })
    print(json.dumps({
        "checkpoint": record["name"],
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "validation_digest": record["corpus"]["val"],
        "drawings": count,
        "device": "cpu",
        "dtype": str(next(checkpoint.model.parameters()).dtype),
        "torch_version": str(torch.__version__),
        "full_bits_per_drawing": float(full.mean()),
        "archived_bits_per_drawing": float(np.mean(record["best_val_bits"])),
        "full_mask_max_abs_error_bits_per_drawing": max_mask_error,
        "windows": windows,
    }, indent=2))


if __name__ == "__main__":
    main()
