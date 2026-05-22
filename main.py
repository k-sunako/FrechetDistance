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
    num_curves: int = 100,
    num_points: int = 200,
    cycles: float = 2.0,
    amplitude: float = 1.0,
    offset_step: float = 0.1,
    phase_step: float = 0.0,
) -> list[list[Point2D]]:
    """
    正弦波をオフセットした2次元曲線を複数生成します。

    各曲線は x 軸方向に 0 から 2pi * cycles までを取り、
    y = amplitude * sin(x + phase) + offset
    で構成します。

    Args:
        num_curves: 生成する曲線数
        num_points: 1曲線あたりの点数
        cycles: x 軸方向に何周期分作るか
        amplitude: 正弦波の振幅
        offset_step: 曲線ごとの y オフセット増分
        phase_step: 曲線ごとの位相差増分

    Returns:
        曲線のリスト
    """
    if num_curves < 1:
        raise ValueError("num_curves は 1 以上で指定してください。")
    if num_points < 2:
        raise ValueError("num_points は 2 以上で指定してください。")

    x_values = np.linspace(0.0, 2.0 * math.pi * cycles, num_points)
    curves: list[list[Point2D]] = []

    for i in range(num_curves):
        offset = i * offset_step
        phase = i * phase_step
        curve = [
            (float(x), float(amplitude * math.sin(x + phase) + offset))
            for x in x_values
        ]
        curves.append(curve)

    return curves


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

    curves = generate_sine_curves(num_curves=100)
    print(f"generated curves: {len(curves)}")
    print(f"first curve points: {len(curves[0])}")
    print(f"last curve points: {len(curves[-1])}")

    plot_curves(curves)


if __name__ == "__main__":
    main()
