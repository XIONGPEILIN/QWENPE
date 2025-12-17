#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计二值 mask 的前景 bbox（紧致包围盒）宽/高/面积分布。

默认会扫描目录下的 `mask_combined_*.png`，并输出：
1) 每张图的 bbox 数据到 CSV
2) 终端打印整体分位数统计

可选：
- `--workers` / `--backend` 多进程/多线程加速
- 如果安装了 matplotlib，可用 `--plot` 输出直方图 PNG。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Iterable


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "需要 numpy。请先安装：pip install numpy"
        ) from e


def _try_import_pil_image():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "需要 Pillow。请先安装：pip install pillow"
        ) from e


@dataclass(frozen=True)
class BBoxStat:
    path: str
    img_w: int
    img_h: int
    empty: bool
    x0: int
    y0: int
    x1: int
    y1: int
    bbox_w: int
    bbox_h: int
    bbox_area: int
    bbox_area_ratio: float


def iter_mask_paths(mask_dir: str, pattern: str) -> Iterable[str]:
    mask_dir = os.path.abspath(mask_dir)
    paths = sorted(glob.glob(os.path.join(mask_dir, pattern)))
    for p in paths:
        if os.path.isfile(p):
            yield p


def compute_bbox_stat(path: str) -> BBoxStat:
    np = _try_import_numpy()
    Image = _try_import_pil_image()

    with Image.open(path) as im:
        # 任何非零都视为前景
        arr = np.asarray(im)
        if arr.ndim == 3:
            # RGB/RGBA：转成单通道（只要存在任何通道非零）
            fg = (arr[..., :3] > 0).any(axis=-1)
        else:
            fg = arr > 0

    img_h, img_w = fg.shape[:2]
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return BBoxStat(
            path=path,
            img_w=img_w,
            img_h=img_h,
            empty=True,
            x0=0,
            y0=0,
            x1=-1,
            y1=-1,
            bbox_w=0,
            bbox_h=0,
            bbox_area=0,
            bbox_area_ratio=0.0,
        )

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    bbox_area = int(bbox_w * bbox_h)
    bbox_area_ratio = float(bbox_area) / float(img_w * img_h)
    return BBoxStat(
        path=path,
        img_w=img_w,
        img_h=img_h,
        empty=False,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        bbox_area=bbox_area,
        bbox_area_ratio=bbox_area_ratio,
    )


def _quantiles(values, qs):
    np = _try_import_numpy()
    if len(values) == 0:
        return {q: None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    out = np.quantile(arr, qs)
    return {float(q): float(v) for q, v in zip(qs, out)}


def print_summary(stats: list[BBoxStat]) -> None:
    np = _try_import_numpy()

    total = len(stats)
    empty = sum(1 for s in stats if s.empty)
    valid = total - empty
    print(f"total={total} valid={valid} empty={empty}")
    if valid == 0:
        return

    bw = [s.bbox_w for s in stats if not s.empty]
    bh = [s.bbox_h for s in stats if not s.empty]
    area = [s.bbox_area for s in stats if not s.empty]
    ratio = [s.bbox_area_ratio for s in stats if not s.empty]

    qs = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    for name, vals in [
        ("bbox_w", bw),
        ("bbox_h", bh),
        ("bbox_area", area),
        ("bbox_area_ratio", ratio),
    ]:
        qv = _quantiles(vals, qs)
        print(
            f"{name}: "
            + " ".join(f"p{int(q*100):02d}={qv[q]:.4g}" for q in qs if qv[q] is not None)
        )

    # 额外给个均值/标准差
    for name, vals in [
        ("bbox_w", bw),
        ("bbox_h", bh),
        ("bbox_area", area),
        ("bbox_area_ratio", ratio),
    ]:
        arr = np.asarray(vals, dtype=np.float64)
        print(f"{name}_mean={arr.mean():.4g} {name}_std={arr.std():.4g}")


def write_csv(stats: list[BBoxStat], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "img_w",
                "img_h",
                "empty",
                "x0",
                "y0",
                "x1",
                "y1",
                "bbox_w",
                "bbox_h",
                "bbox_area",
                "bbox_area_ratio",
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s.path,
                    s.img_w,
                    s.img_h,
                    int(s.empty),
                    s.x0,
                    s.y0,
                    s.x1,
                    s.y1,
                    s.bbox_w,
                    s.bbox_h,
                    s.bbox_area,
                    f"{s.bbox_area_ratio:.10f}",
                ]
            )


def maybe_plot(stats: list[BBoxStat], out_prefix: str, bins: int) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("未安装 matplotlib，跳过绘图（可选安装：pip install matplotlib）")
        return

    valid = [s for s in stats if not s.empty]
    if not valid:
        return

    def _hist(data, title, out_path, logy=False):
        plt.figure(figsize=(7, 5))
        plt.hist(data, bins=bins)
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel("count")
        if logy:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    _hist([s.bbox_w for s in valid], "bbox_w", out_prefix + "_bbox_w.png")
    _hist([s.bbox_h for s in valid], "bbox_h", out_prefix + "_bbox_h.png")
    _hist([s.bbox_area for s in valid], "bbox_area", out_prefix + "_bbox_area.png", logy=True)
    _hist(
        [s.bbox_area_ratio for s in valid],
        "bbox_area_ratio",
        out_prefix + "_bbox_area_ratio.png",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="pico-banana-400k-subject_driven/openimages/ref_gt_generated_all",
        help="mask 目录",
    )
    ap.add_argument(
        "--pattern",
        default="mask_combined_*.png",
        help="glob pattern，比如 mask_combined_*.png",
    )
    ap.add_argument(
        "--out-csv",
        default="bbox_stats.csv",
        help="输出 CSV 路径",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="输出直方图 PNG（需要 matplotlib）",
    )
    ap.add_argument(
        "--plot-prefix",
        default="bbox_hist",
        help="直方图输出前缀（不含扩展名）",
    )
    ap.add_argument("--bins", type=int, default=50, help="直方图 bins")
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="并行 worker 数；0=自动(=CPU核数)，1=不并行",
    )
    ap.add_argument(
        "--backend",
        choices=["process", "thread"],
        default="process",
        help="并行后端：process(默认) / thread",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="并行 map 的 chunksize（仅 process 后端有效）",
    )
    args = ap.parse_args()

    paths = list(iter_mask_paths(args.dir, args.pattern))
    if not paths:
        raise SystemExit(f"未找到任何文件：dir={args.dir} pattern={args.pattern}")

    if args.workers == 0:
        args.workers = os.cpu_count() or 1
    if args.workers < 0:
        raise SystemExit("--workers 不能为负数")

    stats: list[BBoxStat] = []
    total = len(paths)
    if args.workers == 1:
        for i, p in enumerate(paths, 1):
            stats.append(compute_bbox_stat(p))
            if i % 500 == 0:
                print(f"processed {i}/{total}")
    else:
        Executor = ProcessPoolExecutor if args.backend == "process" else ThreadPoolExecutor
        exec_kwargs = {"max_workers": args.workers}
        map_kwargs = {}
        if args.backend == "process":
            map_kwargs["chunksize"] = max(1, int(args.chunksize))
        with Executor(**exec_kwargs) as ex:
            for i, s in enumerate(ex.map(compute_bbox_stat, paths, **map_kwargs), 1):
                stats.append(s)
                if i % 1000 == 0:
                    print(f"processed {i}/{total}")

    write_csv(stats, args.out_csv)
    print_summary(stats)
    print(f"wrote: {args.out_csv}")
    if args.plot:
        maybe_plot(stats, args.plot_prefix, bins=args.bins)
        print(f"wrote: {args.plot_prefix}_*.png")


if __name__ == "__main__":
    main()
