"""What a run was trained on, identified by the data rather than by the config.

Every corpus mix-up in this project so far was found by hand, after the number
had been reported, and each time the config said the two runs matched:

- Tier C's `budget_table` differenced an x18-augmented arm against a plain one
  as if augmentation were a budget, because `(codec, shape, steps, seed)` is not
  unique. The key grew an `augment` term.
- Run 4's planner was compared against an AR baseline differing in *categories*,
  *rdp_eps* and *n_train* at once. The key had grown a categories term by then,
  and still said they matched -- `rdp_eps` lives in `extra`, which is empty when
  the flag defaults, so the same corpus has two keys depending on whether the
  flag was typed; and `n_train` was never in the key at all.

The pattern is that a key derived from a config cannot see a default, cannot see
a corpus the config names indirectly, and grows a term only after something has
already been mis-reported. A hash of the programs can see all of it, because it
is not a description of the corpus -- it *is* the corpus, at 8 bytes.

`train` and `val` are digested separately so a mismatch says which half differs.
That distinction is not cosmetic: two arms on the same val split are comparable
per program even if one trained on more data, and the table should say
"different training set, same measurement" rather than refusing outright.
"""

from __future__ import annotations

import hashlib


def digest(programs: list[bytes]) -> str:
    """A stable 16-hex-character digest of an ordered list of programs.

    Length-prefixed, so `[b"ab", b"c"]` and `[b"a", b"bc"]` cannot collide --
    the corpora this distinguishes differ by exactly that kind of re-splitting
    when an `rdp_eps` changes.

    Order is part of the identity. Two runs that saw the same programs in a
    different order are not the same run: batching follows the list, and
    `_interleave` exists precisely because a `limit=` over a concatenated
    corpus drew all 1,000 val programs from one category.
    """
    h = hashlib.blake2b(digest_size=8)
    for program in programs:
        h.update(len(program).to_bytes(4, "big"))
        h.update(program)
    return h.hexdigest()


def fingerprint(train: list[bytes], val: list[bytes]) -> dict:
    """The record's `corpus` field: both halves, digested and counted.

    Cheap enough to run unconditionally -- blake2b over Tier B's 350k programs
    is ~56 MB and well under a second, against training runs measured in hours.
    """
    return {
        "train": digest(train),
        "val": digest(val),
        "n_train": len(train),
        "n_val": len(val),
        "bytes_train": sum(map(len, train)),
        "bytes_val": sum(map(len, val)),
    }
