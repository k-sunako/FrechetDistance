from __future__ import annotations

from typing import Sequence

import numpy as np


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


if __name__ == "__main__":
    main()
