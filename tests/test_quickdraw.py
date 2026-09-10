

def test_interleave_makes_every_prefix_balanced():
    """The bug this guards: `load(limit=n)` slices from the front, so a corpus
    concatenated by category is single-category for any n below one category's
    size -- which would have made train and val different distributions."""
    from dm.data.quickdraw import _interleave

    groups = [[bytes([c]) * 1 for _ in range(10)] for c in (1, 2, 3)]
    out = _interleave(groups)
    assert len(out) == 30
    for n in (3, 6, 9, 30):
        prefix = out[:n]
        counts = {c: sum(p == bytes([c]) for p in prefix) for c in (1, 2, 3)}
        assert set(counts.values()) == {n // 3}, (n, counts)


def test_interleave_handles_unequal_categories():
    from dm.data.quickdraw import _interleave

    out = _interleave([[b"a"] * 2, [b"b"] * 5])
    assert out == [b"a", b"b", b"a", b"b", b"b", b"b", b"b"]
