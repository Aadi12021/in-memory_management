from __future__ import annotations

from memory_system.events import ConsolidationReport


def test_default_construction_has_empty_lists():
    report = ConsolidationReport()
    assert report.merged == []
    assert report.compressed == []
    assert report.strengthened == []


def test_constructs_with_populated_fields():
    report = ConsolidationReport(
        merged=[("a", "b", "c")],
        compressed=[(["d", "e"], "f")],
        strengthened=[("g", "h")],
    )
    assert report.merged == [("a", "b", "c")]
    assert report.compressed == [(["d", "e"], "f")]
    assert report.strengthened == [("g", "h")]


def test_dry_run_new_id_can_be_none():
    report = ConsolidationReport(merged=[("a", "b", None)], compressed=[(["c"], None)])
    assert report.merged[0][2] is None
    assert report.compressed[0][1] is None
