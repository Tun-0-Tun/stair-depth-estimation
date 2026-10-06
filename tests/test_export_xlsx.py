"""export_xlsx picks the right rows per sheet and computes paper deviations with formulas."""

from __future__ import annotations

import csv
import sys

import pytest

openpyxl = pytest.importorskip("openpyxl")
sys.path.insert(0, "scripts")

from export_xlsx import build, load_rows  # noqa: E402

from utils.results import SUMMARY_COLUMNS  # noqa: E402


def _row(ts, exp, ds, method, rmse, **kw):
    r = dict.fromkeys(SUMMARY_COLUMNS, "")
    r.update(
        timestamp=ts,
        experiment=exp,
        dataset=ds,
        method=method,
        rmse=rmse,
        n_samples=10,
        degradation="none",
    )
    r.update(kw)
    return r


def test_sheets_pick_their_rows_dedupe_and_compare_with_the_paper(tmp_path):
    rows = [
        _row("20261001-000000", "ducos_middlebury_x4", "middlebury", "ducos", 0.0100),  # stale
        _row("20261005-000000", "ducos_middlebury_x4", "middlebury", "ducos", 0.0057),
        _row("20261005-000001", "bicubic_middlebury_x4", "middlebury", "bicubic", 0.0090),
        _row("20261005-000002", "wave_nyuv2_x8", "nyuv2", "wave", 0.0250),
        _row("20261005-000003", "bicubic_nyuv2_x8", "nyuv2", "bicubic", 0.0717),
        _row(
            "20261005-000004", "wave_void_stairs_x8", "void_stairs", "wave", 9.0
        ),  # invalid protocol
        _row("20261005-000005", "depthor_zju_l5", "zju_l5", "depthor", 0.3501),
        _row("20261005-000006", "adhoc", "zju_l5", "nn_fill", 0.5775),
        _row(
            "20261005-000007", "adhoc", "minjiang", "ducos", 9.0
        ),  # old matrix run, must not appear
    ]
    path = tmp_path / "benchmark.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS))
        w.writeheader()
        w.writerows(rows)

    wb, missing = build(load_rows(path), {"wave_minjiang_x8", "ducos_middlebury_x4"})
    exps = {ws.title: [c.value for c in ws["A"][5:] if c.value] for ws in wb}
    assert exps["DuCos"] == ["ducos_middlebury_x4", "bicubic_middlebury_x4"], (
        "latest row, neural first"
    )
    assert exps["WAVE"] == ["wave_nyuv2_x8", "bicubic_nyuv2_x8"], (
        "invalid wave_void_stairs left out"
    )
    assert exps["DEPTHOR"] == ["depthor_zju_l5", "zju_l5/nn_fill"]
    assert missing["WAVE"] == ["wave_minjiang_x8"]

    ws = wb["DuCos"]
    head = {c.value: c.column_letter for c in ws[5]}
    r = 6
    assert (
        ws[f"{head['Наш RMSE в ед. статьи']}{r}"].value
        == f"={head['RMSE ↓']}{r}*{head['× к ед. статьи']}{r}"
    )
    assert ws[f"{head['RMSE по статье']}{r}"].value == 1.45
