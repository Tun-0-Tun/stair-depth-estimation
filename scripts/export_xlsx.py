#!/usr/bin/env python3
"""Export ``benchmark.csv`` to one .xlsx for sharing: one sheet per neural method.

    uv run --with openpyxl python scripts/export_xlsx.py
    uv run --with openpyxl python scripts/export_xlsx.py --out results.xlsx

Each sheet holds the method's rows plus the no-training reference rows that run on
the same protocol, every metric, and -- where the paper reports one -- the paper's
RMSE next to ours in the paper's units.  The latest row per experiment is used, so
re-running after new experiments just refreshes the file.  Experiments that exist
as configs but have no row are listed on stdout.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from utils.config import CONFIG_ROOT
from utils.results import RESULTS_DIR, migrate_summary_header

# sheet -> (neural method key, experiment-name regex of every row that belongs to it)
SHEETS: dict[str, tuple[str, str]] = {
    "DEPTHOR": (
        "depthor",
        r"^(depthor|nn_fill|local_bilateral)_",
    ),
    "DuCos": (
        "ducos",
        r"^(ducos_|bicubic_.*_x4$|bicubic_arkitscenes$)",
    ),
    # wave_zju_l5_* / wave_void_stairs_* are invalid protocols (metrics_protocol.md)
    "WAVE": (
        "wave",
        r"^(wave_(minjiang|nyuv2|rgbd_stair|arkitscenes)|bicubic_.*_x(8|16|32)$|bicubic_arkitscenes$)",
    ),
}

# dataset -> (stairs?, units, what the ground truth is)
DATASETS: dict[str, tuple[str, str, str]] = {
    "zju_l5": ("нет", "м", "стерео-реконструкция"),
    "minjiang": ("да", "м", "сам датчик (оптимистично)"),
    "rgbd_stair": ("да", "м", "сам датчик, 8 бит (оптимистично)"),
    "void_stairs": ("да", "м", "плотный эталон VOID"),
    "hammer": ("нет", "м", "лазер"),
    "arkitscenes": ("нет", "м", "лазер"),
    "nyuv2": ("нет", "м", "Kinect, заполненная глубина"),
    "middlebury": ("нет", "норм. 0-1, НЕ метры", "структурированный свет"),
    "lu": ("нет", "норм. 0-1, НЕ метры", "ASUS Xtion"),
}

# (sheet, experiment) -> (paper RMSE, factor ours->paper units, source)
PAPER: dict[tuple[str, str], tuple[float, float, str]] = {
    ("DEPTHOR", "depthor_zju_l5"): (
        0.350,
        1,
        "DEPTHOR, arXiv:2504.01596, Tab. 2 (Ours-Large), ZJU-L5, м",
    ),
    ("DuCos", "ducos_middlebury_x4"): (
        1.45,
        255,
        "DuCos, arXiv:2503.04171, Tab. 1, x4; шкала 0-255",
    ),
    ("DuCos", "ducos_lu_x4"): (1.38, 255, "DuCos, arXiv:2503.04171, Tab. 1, x4; шкала 0-255"),
    ("DuCos", "ducos_nyuv2_x4"): (2.60, 100, "DuCos, arXiv:2503.04171, Tab. 1, x4; см"),
    ("WAVE", "wave_nyuv2_x8"): (2.50, 100, "WAVE, arXiv:2608.25302, Tab. 3 (обучен на NYU); см"),
    ("WAVE", "wave_nyuv2_x16"): (4.60, 100, "WAVE, arXiv:2608.25302, Tab. 3 (обучен на NYU); см"),
    ("WAVE", "wave_nyuv2_x32"): (7.90, 100, "WAVE, arXiv:2608.25302, Tab. 1 (обучен на NYU); см"),
}

# (header, csv field, number format); ↓ lower is better, ↑ higher
METRICS = [
    ("AbsRel ↓", "absrel", "0.0000"),
    ("RMSE ↓", "rmse", "0.0000"),
    ("RMSE(log) ↓", "rmse_log", "0.0000"),
    ("δ1 ↑", "delta1", "0.0000"),
    ("δ2 ↑", "delta2", "0.0000"),
    ("δ3 ↑", "delta3", "0.0000"),
    ("MAE ↓", "mae", "0.0000"),
    ("SqRel ↓", "sqrel", "0.0000"),
    ("SiLog ↓", "silog", "0.00"),
    ("FPS ↑", "fps", "0.0"),
    ("Задержка, мс ↓", "latency_ms_median", "0.0"),
]
HEAD = (
    ["Эксперимент", "Датасет", "Лестницы", "Вход", "Эталон", "Метод", "Роль", "Кадров"]
    + [m[0] for m in METRICS]
    + ["Ед. RMSE", "RMSE по статье", "× к ед. статьи", "Наш RMSE в ед. статьи", "Откл., %"]
    + ["Примечание", "Устройство", "Коммит", "Дата"]
)
FONT = "Arial"


def load_rows(path: Path) -> list[dict[str, str]]:
    """benchmark.csv under the current header, without touching the original."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "benchmark.csv"
        shutil.copy(path, copy)
        migrate_summary_header(copy)
        with copy.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


def latest_per_experiment(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Newest row per experiment name; matrix runs ('adhoc') are keyed by dataset/method."""
    out: dict[str, dict[str, str]] = {}
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        key = r["experiment"]
        if key == "adhoc":
            if r["dataset"] != "zju_l5" or r["method"] not in ("nn_fill", "bicubic"):
                continue  # the only matrix rows we keep: floors on ZJU-L5, beside DEPTHOR
            key = f"{r['dataset']}/{r['method']}"
        out[key] = r
    return out


def protocol(sheet: str, exp: str, r: dict[str, str]) -> str:
    ds = r["dataset"]
    if sheet == "DEPTHOR":
        if r["degradation"] == "dtof_sim":
            src = "датчика" if ds in ("hammer", "arkitscenes") else "эталона"
            return f"симул. dToF 8×8 из {src}"
        return "~740 точек VIO" if ds == "void_stairs" else "реальный dToF 8×8"
    m = re.search(r"_x(\d+)$", exp)
    if m:
        return f"bicubic ×{m.group(1)} из эталона"
    return "реальный LiDAR 256×192 (×7.5)"


def sort_key(sheet: str, r: dict[str, str], exp: str):
    order = list(DATASETS)
    scale = re.search(r"_x(\d+)$", exp)
    neural = SHEETS[sheet][0] == r["method"]
    return (
        order.index(r["dataset"]) if r["dataset"] in order else 99,
        int(scale.group(1)) if scale else 0,
        not neural,
        r["method"],
    )


def num(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build(rows: list[dict[str, str]], configs: set[str]) -> tuple[Workbook, dict[str, list[str]]]:
    latest = latest_per_experiment(rows)
    wb = Workbook()
    wb.remove(wb.active)
    missing: dict[str, list[str]] = {}
    thin = Side(style="thin", color="999999")
    notes = {
        "DEPTHOR": "FPS DEPTHOR измерен через bpops_shim (замена CUDA-расширения на PyTorch): скорость с DuCos/WAVE не сравнивать.",
        "DuCos": "Все ячейки DuCos - ×4 (один чекпойнт). Middlebury/Lu: RMSE в шкале 0-1; ×255 = число из статьи.",
        "WAVE": "WAVE: по чекпойнту на масштаб (×8/×16/×32). Лестничные датасеты: эталон - сам датчик, цифры оптимистичны.",
    }
    for sheet, (method, pattern) in SHEETS.items():
        ws = wb.create_sheet(sheet)
        ws["A1"] = f"{sheet}: все метрики"
        ws["A1"].font = Font(name=FONT, bold=True, size=12)
        ws["A2"] = notes[sheet]
        ws["A3"] = (
            "Голубым выделена сама нейросеть; остальные строки - ориентиры без обучения под задачу, на том же входе. "
            "↓ - меньше лучше, ↑ - больше лучше."
        )
        for c in ("A2", "A3"):
            ws[c].font = Font(name=FONT, size=9, italic=True)
        head_row = 5
        for j, h in enumerate(HEAD, 1):
            c = ws.cell(head_row, j, h)
            c.font = Font(name=FONT, bold=True, size=10)
            c.fill = PatternFill("solid", fgColor="D9D9D9")
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            c.border = Border(bottom=Side(style="medium"))
        col = {h: j for j, h in enumerate(HEAD, 1)}

        # "dataset/method" keys are the matrix floors on ZJU-L5, which sit beside DEPTHOR
        picked = sorted(
            (
                (e, r)
                for e, r in latest.items()
                if re.search(pattern, e) or (sheet == "DEPTHOR" and "/" in e)
            ),
            key=lambda kv: sort_key(sheet, kv[1], kv[0]),
        )
        missing[sheet] = sorted(e for e in configs if re.search(pattern, e) and e not in latest)

        prev_ds, row_i = None, head_row + 1
        for exp, r in picked:
            ds = r["dataset"]
            stairs, units, gt = DATASETS.get(ds, ("", "", ""))
            neural = r["method"] == method
            vals = [
                exp,
                ds,
                stairs,
                protocol(sheet, exp, r),
                gt,
                r["method"],
                "нейросеть" if neural else "ориентир",
                num(r["n_samples"]),
            ]
            vals += [num(r[m[1]]) for m in METRICS]
            vals += [units]
            for j, v in enumerate(vals, 1):
                c = ws.cell(row_i, j, v)
                c.font = Font(name=FONT, size=10, bold=neural and j in (1, 6))
                if neural:
                    c.fill = PatternFill("solid", fgColor="DDEBF7")
                if prev_ds is not None and ds != prev_ds:
                    c.border = Border(top=thin)
            for name, _, fmt in METRICS:
                ws.cell(row_i, col[name]).number_format = fmt
            ws.cell(row_i, col["Кадров"]).number_format = "#,##0"

            paper = PAPER.get((sheet, exp))
            if paper:
                p_val, factor, src = paper
                rmse_cell = f"{get_column_letter(col['RMSE ↓'])}{row_i}"
                pc, fc = col["RMSE по статье"], col["× к ед. статьи"]
                ws.cell(row_i, pc, p_val).font = Font(name=FONT, size=10, color="0000FF")
                ws.cell(row_i, pc).comment = Comment(src, "export_xlsx")
                ws.cell(row_i, fc, factor).font = Font(name=FONT, size=10, color="0000FF")
                ours, p_ref, f_ref = (
                    col["Наш RMSE в ед. статьи"],
                    f"{get_column_letter(pc)}{row_i}",
                    f"{get_column_letter(fc)}{row_i}",
                )
                ws.cell(row_i, ours, f"={rmse_cell}*{f_ref}").number_format = "0.000"
                ws.cell(
                    row_i, col["Откл., %"], f"=({get_column_letter(ours)}{row_i}-{p_ref})/{p_ref}"
                ).number_format = "0.0%"
                for name in ("Наш RMSE в ед. статьи", "Откл., %"):
                    ws.cell(row_i, col[name]).font = Font(name=FONT, size=10)
                if neural:
                    for name in (
                        "RMSE по статье",
                        "× к ед. статьи",
                        "Наш RMSE в ед. статьи",
                        "Откл., %",
                    ):
                        ws.cell(row_i, col[name]).fill = PatternFill("solid", fgColor="DDEBF7")

            tail = [
                r["notes"],
                r["device"],
                r["git_commit"],
                f"{r['timestamp'][:4]}-{r['timestamp'][4:6]}-{r['timestamp'][6:8]}",
            ]
            for k, v in enumerate(tail):
                c = ws.cell(row_i, col["Примечание"] + k, v)
                c.font = Font(name=FONT, size=9)
            prev_ds, row_i = ds, row_i + 1

        widths = {
            "Эксперимент": 30,
            "Датасет": 12,
            "Лестницы": 9,
            "Вход": 26,
            "Эталон": 28,
            "Метод": 16,
            "Роль": 11,
            "Кадров": 8,
            "Ед. RMSE": 18,
            "Примечание": 48,
            "Устройство": 9,
            "Коммит": 16,
            "Дата": 11,
        }
        for h, j in col.items():
            ws.column_dimensions[get_column_letter(j)].width = widths.get(h, 11)
        ws.row_dimensions[head_row].height = 42
        ws.freeze_panes = ws.cell(head_row + 1, 3)
        ws.auto_filter.ref = (
            f"A{head_row}:{get_column_letter(len(HEAD))}{max(row_i - 1, head_row + 1)}"
        )
    return wb, missing


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", type=Path, default=RESULTS_DIR / "benchmark.csv")
    p.add_argument(
        "--out", type=Path, default=RESULTS_DIR / f"baselines_{time.strftime('%Y-%m-%d')}.xlsx"
    )
    args = p.parse_args()

    rows = load_rows(args.csv)
    configs = {
        q.stem for q in (CONFIG_ROOT / "experiment").glob("*.yaml") if not q.stem.startswith("_")
    }
    wb, missing = build(rows, configs)
    wb.save(args.out)
    print(f"{len(rows)} rows read, wrote {args.out}")
    for sheet, names in missing.items():
        if names:
            print(f"[{sheet}] no result yet for: {' '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
