"""The corpus a run was scored on, identified rather than described.

Three corpus mix-ups reached a reported number in this project, and every one
of them was found by hand afterwards. What they share is that the config said
the runs matched: `budget_table` differenced an x18-augmented Tier C arm
against a plain one, and run 4's planner was compared against an AR baseline
differing in categories, `rdp_eps` and `n_train` at once.

So these tests are about what a *config-derived* key cannot see, and what a
digest of the programs can.
"""

from dm.data.fingerprint import digest, fingerprint


def test_a_defaulted_flag_and_a_typed_one_are_one_corpus():
    """`--rdp-eps 2.0` typed and omitted produce identical programs, and the
    config-derived key gave them different names because `extra` is empty when
    the flag defaults. The digest cannot tell the two invocations apart,
    because there is nothing to tell apart -- it sees the data."""
    programs = [b"\x01\x10\x20", b"\x01\x30\x40\x00"]
    assert digest(programs) == digest(list(programs))


def test_the_digest_separates_corpora_a_config_key_called_equal():
    """Run 4's case, in miniature: same categories, same everything the key
    looked at, different programs because `rdp_eps` moved."""
    coarse = [b"\x01\x10\x20", b"\x01\x30\x40"]
    fine = [b"\x01\x10\x20\x11\x18\x28", b"\x01\x30\x40"]
    assert digest(coarse) != digest(fine)


def test_length_prefixing_stops_a_resplit_from_colliding():
    """The corpora this has to separate differ by exactly this kind of
    re-splitting: an `rdp_eps` change moves byte boundaries between programs
    while leaving the concatenation nearly alone. Without the length prefix
    those hash the same."""
    assert digest([b"ab", b"c"]) != digest([b"a", b"bc"])


def test_order_is_part_of_the_corpus():
    """`quickdraw.load` appended category after category, so a `limit=` over
    five categories drew all 1,000 val programs from the first one;
    `_interleave` exists to make any prefix balanced. Two runs that saw the
    same programs in a different order did not see the same corpus."""
    assert digest([b"\x01", b"\x02"]) != digest([b"\x02", b"\x01"])


def test_the_two_halves_are_separable():
    """A mismatch has to say *which* half differs. Two arms on one val split
    are comparable per program even when one trained on more data -- that is a
    statement the table should make, not a reason to refuse the pair. Run 4
    differed in both halves and the record could say neither."""
    train, val = [b"\x01\x10\x20"], [b"\x01\x30\x40"]
    same_val = fingerprint(train + [b"\x01\x50\x60"], val)
    base = fingerprint(train, val)
    assert base["val"] == same_val["val"]
    assert base["train"] != same_val["train"]
    assert base["n_train"] == 1 and same_val["n_train"] == 2


def test_the_fingerprint_reports_what_a_mismatch_needs_to_be_explained():
    programs = [b"\x01\x10\x20", b"\x01\x30\x40\x00"]
    marks = fingerprint(programs, programs[:1])
    assert marks["n_train"] == 2 and marks["n_val"] == 1
    assert marks["bytes_train"] == 7 and marks["bytes_val"] == 3
