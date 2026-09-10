"""A typed word to one of the checkpoint's classes, or an honest refusal.

The demo asks for a word and the model knows five categories. Those two facts
have to meet somewhere, and *where* they meet is a claim about the system: a
router that silently maps every input to its nearest class is a demo that
implies open-vocabulary capability the 825k-parameter model does not have, and
that is the same failure mode as reporting an unconverged delta as a result.

So the mapping is explicit and visible. Three outcomes, and the page shows which
one fired:

- **exact** -- the word *is* a class the checkpoint was trained on;
- **synonym** -- the word is in a hand-written table below, and the page prints
  the substitution it made;
- **unknown** -- refused, naming the entire vocabulary.

The synonym table is deliberately small and hand-written rather than an
embedding lookup. A sentence-embedding model would route `tiger` to `cat` with
a similarity score, which reads as capability and is really the *embedding
model's* knowledge standing in for the drawing model's. Nothing in this project
gets to borrow evidence from a component it did not measure.

`classes` comes from the checkpoint's own config rather than a constant here.
A router that hardcodes the five names is a router that keeps working, wrongly,
against a checkpoint trained on a different five -- and `docs/claim4-bringup.md`
§11.2 already made exactly that mistake in prose, naming `cat/dog/bus/car/tree`
against a checkpoint trained on `cat/bus/flower/sailboat/bicycle`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Alternative spellings, per class name. A word may appear under exactly one
#: class -- `_check_disjoint` enforces it at import, because a table where
#: `boat` reaches both `sailboat` and something else would resolve by dict
#: order, which is not a decision anyone made.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "cat": ("kitten", "kitty", "feline", "tomcat", "gato", "chat", "katze"),
    "bus": ("coach", "minibus", "schoolbus", "autobus", "omnibus", "autobús"),
    "flower": ("blossom", "bloom", "rose", "daisy", "tulip", "flor", "fleur"),
    "sailboat": ("boat", "sailing", "sailingboat", "yacht", "dinghy", "ship",
                 "velero", "voilier"),
    "bicycle": ("bike", "bicicleta", "cycle", "pushbike", "velo", "vélo",
                "fahrrad"),
}

#: Suffixes stripped before lookup, longest first. English plurals only: this is
#: a spelling normaliser, not a stemmer, and an aggressive one would map
#: `busses` and `bussing` to the same place while `flowering` quietly became
#: `flower`.
_PLURAL = ("es", "s")

_PUNCTUATION = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_SPACES = re.compile(r"[\s_-]+")


def _check_disjoint() -> None:
    seen: dict[str, str] = {}
    for klass, words in SYNONYMS.items():
        for word in (klass, *words):
            if word in seen and seen[word] != klass:
                raise ValueError(
                    f"{word!r} routes to both {seen[word]!r} and {klass!r}; a "
                    "synonym table that resolves by dict order is not a decision"
                )
            seen[word] = klass


_check_disjoint()


def normalise(word: str) -> str:
    """Lowercase, unaccented, punctuation-free, single-token.

    Accents are folded so `vélo` and `velo` are one entry rather than two that
    can drift apart, and the table above keeps both spellings anyway so a
    reader can see which languages were considered.
    """
    text = unicodedata.normalize("NFKD", word.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCTUATION.sub("", text)
    return _SPACES.sub("", text)


@dataclass(frozen=True)
class Routing:
    """What the word became, and by which rule."""

    word: str
    #: `exact`, `synonym` or `unknown`.
    rule: str
    #: The class name, or None when refused.
    klass: str | None
    #: The class index the model conditions on, or None when refused.
    index: int | None
    #: One sentence for the page. Always shown, including on the exact path,
    #: so a viewer never has to wonder whether a substitution happened.
    reason: str

    @property
    def routed(self) -> bool:
        return self.index is not None

    def as_dict(self) -> dict:
        return {
            "word": self.word,
            "rule": self.rule,
            "class": self.klass,
            "index": self.index,
            "reason": self.reason,
        }


def route(word: str, classes: tuple[str, ...]) -> Routing:
    """Resolve `word` against the checkpoint's own class list.

    `classes` is ordered, and the index is the model's conditioning input, so
    the order is load-bearing: it is `config["categories"]` from the run record
    and must never be sorted, deduplicated or reordered on the way here.
    """
    shown = word.strip()
    key = normalise(word)
    if not key:
        return Routing(word, "unknown", None, None,
                       "no word given; " + _vocabulary_sentence(classes))

    lookup = {normalise(name): name for name in classes}
    for klass, words in SYNONYMS.items():
        if klass not in classes:
            # A synonym for a class this checkpoint was not trained on must not
            # resolve. The table is written for the five-category run and the
            # checkpoint is the authority on what exists.
            continue
        for alias in words:
            lookup.setdefault(normalise(alias), klass)

    for candidate in _candidates(key):
        name = lookup.get(candidate)
        if name is None:
            continue
        index = classes.index(name)
        if candidate == normalise(name):
            return Routing(word, "exact", name, index,
                           f"“{shown}” is category {index} of {len(classes)}.")
        return Routing(
            word, "synonym", name, index,
            f"“{shown}” is not a trained category; drawn as “{name}”, "
            f"category {index} of {len(classes)}.",
        )

    return Routing(word, "unknown", None, None,
                   f"“{shown}” is outside this model's vocabulary. "
                   + _vocabulary_sentence(classes))


def _candidates(key: str) -> tuple[str, ...]:
    """The normalised word, then its singular forms, in that order."""
    out = [key]
    for suffix in _PLURAL:
        if key.endswith(suffix) and len(key) > len(suffix) + 1:
            out.append(key[: -len(suffix)])
    return tuple(out)


def _vocabulary_sentence(classes: tuple[str, ...]) -> str:
    """The refusal text, which states the *whole* vocabulary rather than hinting.

    The five categories are not a demo restriction that a larger run would
    lift; they are what this checkpoint was trained on, and a viewer who is
    told the boundary can read every other number on the page against it.
    """
    listed = ", ".join(classes)
    return (f"This model was trained on {len(classes)} QuickDraw categories and "
            f"they are the entire vocabulary: {listed}.")
