"""export_xlsx: one sheet per dataset, all methods on it, paper deviations as formulas."""

from __future__ import annotations

import csv
import sys

import pytest

openpyxl = pytest.importorskip("openpyxl")
sys.path.insert(0, "scripts")

from export_xlsx import HEAD, build, load_rows  # noqa: E402

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


def test_each_dataset_sheet_lists_every_method_on_it(tmp_path):
    rows = [
        _row("20261001-000000", "ducos_middlebury_x4", "middlebury", "ducos", 0.0100),  # stale
        _row(
            "20261005-000000",
            "ducos_middlebury_x4",
            "middlebury",
            "ducos",
            0.0057,
            degradation="bicubic_sr",
        ),
        _row(
            "20261005-000001",
            "bicubic_middlebury_x4",
            "middlebury",
            "bicubic",
            0.0090,
            degradation="bicubic_sr",
        ),
        _row("20261005-000002", "wave_nyuv2_x8", "nyuv2", "wave", 0.0250, degradation="bicubic_sr"),
        _row(
            "20261005-000003", "ducos_nyuv2_x4", "nyuv2", "ducos", 0.0260, degradation="bicubic_sr"
        ),
        _row(
            "20261005-000004",
            "bicubic_nyuv2_x8",
            "nyuv2",
            "bicubic",
            0.0717,
            degradation="bicubic_sr",
        ),
        _row(
            "20261005-000005", "wave_void_stairs_x8", "void_stairs", "wave", 9.0
        ),  # invalid protocol
        _row("20261005-000006", "depthor_zju_l5", "zju_l5", "depthor", 0.3501),
        _row("20261005-000007", "adhoc", "zju_l5", "nn_fill", 0.5775),
        _row(
            "20261005-000008", "adhoc", "minjiang", "ducos", 9.0
        ),  # old matrix run, must not appear
    ]
    path = tmp_path / "benchmark.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS))
        w.writeheader()
        w.writerows(rows)

    wb, missing = build(load_rows(path), {"wave_minjiang_x8", "ducos_middlebury_x4"})
    head = {
        c.value: c.column_letter
        for c in next(r for r in wb["Lu"].iter_rows() if r[0].value == HEAD[0])
    }

    def table(sheet):
        ws = wb[sheet]
        start = next(r[0].row for r in ws.iter_rows() if r[0].value == HEAD[0])
        return {ws[f"A{i}"].value: i for i in range(start + 1, ws.max_row + 1) if ws[f"A{i}"].value}

    assert list(table("Middlebury")) == ["ducos_middlebury_x4", "bicubic_middlebury_x4"], (
        "latest row, neural first"
    )
    assert list(table("NYUv2")) == ["ducos_nyuv2_x4", "wave_nyuv2_x8", "bicubic_nyuv2_x8"], (
        "×4 group before ×8"
    )
    assert list(table("ZJU-L5")) == ["depthor_zju_l5", "zju_l5/nn_fill"]
    assert list(table("VOID")) == ["результатов пока нет"], (
        "wave_void_stairs_x8 is an invalid protocol"
    )
    assert list(table("MinJiang")) == ["результатов пока нет"], "old matrix runs are not exported"
    assert missing == ["wave_minjiang_x8"]
    assert "Примечание" not in HEAD

    ws, r = wb["Middlebury"], table("Middlebury")["ducos_middlebury_x4"]
    assert (
        ws[f"{head['Наш RMSE в ед. статьи']}{r}"].value
        == f"={head['RMSE ↓']}{r}*{head['× к ед. статьи']}{r}"
    )
    assert ws[f"{head['RMSE по статье']}{r}"].value == 1.45
