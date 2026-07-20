"""Regression tests for the silent-e stemming bug found while building
benchmark/retrieval_benchmark.py: stripping -es/-ed/-ing left the
silent 'e' dropped (hike/hiking -> "hike" vs "hik") instead of restored,
so these word pairs didn't stem to the same token.
"""

from memory_system.backends.memory import _stem


def test_hike_hiking_collapse_to_same_stem():
    assert _stem("hike") == _stem("hiking")


def test_live_lives_lived_collapse_to_same_stem():
    assert _stem("live") == _stem("lives") == _stem("lived")


def test_like_likes_collapse_to_same_stem():
    assert _stem("like") == _stem("likes")


def test_love_loves_collapse_to_same_stem():
    assert _stem("love") == _stem("loves")


def test_regular_plurals_and_non_silent_e_verbs_still_collapse():
    """Guard against the fix breaking cases that already worked."""
    assert _stem("peanut") == _stem("peanuts")
    assert _stem("nut") == _stem("nuts")
    assert _stem("allergy") == _stem("allergies")
    assert _stem("lunch") == _stem("lunches")
    assert _stem("walk") == _stem("walking") == _stem("walked")
    assert _stem("train") == _stem("training")
    assert _stem("work") == _stem("worked")
    assert _stem("mention") == _stem("mentioned")


def test_irregular_derivation_still_not_handled():
    """Known, documented limitation, not a regression target: allergic/
    allergy is a different root entirely, not a suffix inflection --
    this stemmer was never meant to catch it.
    """
    assert _stem("allergic") != _stem("allergy")
