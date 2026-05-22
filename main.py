from __future__ import annotations

from typing import Sequence

import math
import numpy as np
import matplotlib.pyplot as plt


Point2D = tuple[float, float]


def _as_point_array(curve: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("曲線は shape=(N, 2) の2次元座標列で指定してください。")
    if arr.shape[0] < 1:
        raise ValueError("曲線は少なくとも1点以上必要です。")
    return arr


def discrete_frechet_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
) -> float:
    """
    2次元曲線同士の discrete Fréchet 距離を計算します。

    入力:
        curve1, curve2: [(x, y), ...] のような点列

    戻り値:
        discrete Fréchet 距離
    """
    p = _as_point_array(curve1)
    q = _as_point_array(curve2)

    n, m = len(p), len(q)
    ca = np.full((n, m), -1.0, dtype=float)

    def c(i: int, j: int) -> float:
        if ca[i, j] >= 0:
            return ca[i, j]

        dist = float(np.linalg.norm(p[i] - q[j]))

        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i == 0:
            ca[i, j] = max(c(i, j - 1), dist)
        elif j == 0:
            ca[i, j] = max(c(i - 1, j), dist)
        else:
            ca[i, j] = max(
                min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                dist,
            )
        return ca[i, j]

    return c(n - 1, m - 1)


def generate_sine_curves(
    num_curves: int = 10,
    num_points: int = 200,
    amplitude: float = 1.0,
    offset_step: float = 0.1,
    phase_step: float = 0.0,
    x_start_min: float = 0.0,
    x_start_max: float = 10.0,
    seed: int | None = None,
) -> list[list[Point2D]]:
    """
    正弦波をオフセットした2次元曲線を複数生成します。

    各曲線は x 軸方向に長さ 2pi の区間を持ち、
    開始位置は曲線ごとにランダムに変わります。

    それぞれの曲線は以下で生成します:
        x in [x_start, x_start + 2pi]
        y = amplitude * sin(x + phase) + offset

    Args:
        num_curves: 生成する曲線数
        num_points: 1曲線あたりの点数
        amplitude: 正弦波の振幅
        offset_step: 曲線ごとの y オフセット増分
        phase_step: 曲線ごとの位相差増分
        x_start_min: ランダムな x 開始位置の最小値
        x_start_max: ランダムな x 開始位置の最大値
        seed: 乱数シード

    Returns:
        曲線のリスト
    """
    if num_curves < 1:
        raise ValueError("num_curves は 1 以上で指定してください。")
    if num_points < 2:
        raise ValueError("num_points は 2 以上で指定してください。")
    if x_start_min > x_start_max:
        raise ValueError("x_start_min は x_start_max 以下で指定してください。")

    rng = np.random.default_rng(seed)
    curves: list[list[Point2D]] = []

    for i in range(num_curves):
        x_start = float(rng.uniform(x_start_min, x_start_max))
        x_end = x_start + 2.0 * math.pi
        x_values = np.linspace(x_start, x_end, num_points)

        offset = i * offset_step
        phase = i * phase_step
        curve = [
            (float(x), float(amplitude * math.sin(x + phase) + offset))
            for x in x_values
        ]
        curves.append(curve)

    return curves


def compute_all_pair_frechet_distances(
    curves: list[list[Point2D]],
) -> list[tuple[int, int, float]]:
    """
    曲線群を総当たりで比較し、全組み合わせの discrete Fréchet 距離を返します。

    Returns:
        (i, j, distance) のリスト
    """
    if not curves:
        raise ValueError("比較する曲線がありません。")

    results: list[tuple[int, int, float]] = []
    n = len(curves)

    for i in range(n):
        for j in range(i + 1, n):
            dist = discrete_frechet_distance(curves[i], curves[j])
            results.append((i, j, dist))

    return results


def plot_curves(curves: list[list[Point2D]]) -> None:
    """
    曲線群を matplotlib で表示します。
    各曲線は異なる色で描画されます。
    """
    if not curves:
        raise ValueError("表示する曲線がありません。")

    cmap = plt.get_cmap("viridis", len(curves))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, curve in enumerate(curves):
        arr = np.asarray(curve, dtype=float)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=cmap(i),
            linewidth=1.5,
            label=f"curve {i + 1}" if i < 10 else None,
        )

    ax.set_title("Generated Sine Curves")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    if len(curves) <= 10:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # 実験用サンプル
    curve_a = [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 0.0),
    ]
    curve_b = [
        (0.0, 0.0),
        (1.0, 0.2),
        (2.0, 0.0),
    ]

    dist = discrete_frechet_distance(curve_a, curve_b)
    print(f"discrete Fréchet distance: {dist:.6f}")

    curves = generate_sine_curves(num_curves=10, seed=42)
    print(f"generated curves: {len(curves)}")
    print(f"first curve points: {len(curves[0])}")
    print(f"last curve points: {len(curves[-1])}")

    pairwise_distances = compute_all_pair_frechet_distances(curves)
    print(f"pairwise comparisons: {len(pairwise_distances)}")
    for i, j, d in pairwise_distances[:10]:
        print(f"curve[{i}] vs curve[{j}]: {d:.6f}")

    plot_curves(curves)


if __name__ == "__main__":
    main()
