from __future__ import annotations

from typing import Sequence

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


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


def rotate_curve(
    curve: Sequence[Sequence[float]],
    angle_degrees: float,
    center: Sequence[float] | None = None,
) -> list[Point2D]:
    """
    曲線を指定角度だけ回転させます。

    Args:
        curve: 回転対象の2次元曲線
        angle_degrees: 回転角度（度）
        center: 回転中心。None の場合は曲線の重心を使う

    Returns:
        回転後の曲線
    """
    arr = _as_point_array(curve)

    if center is None:
        cx = float(arr[:, 0].mean())
        cy = float(arr[:, 1].mean())
    else:
        center_arr = np.asarray(center, dtype=float)
        if center_arr.shape != (2,):
            raise ValueError("center は長さ2の座標で指定してください。")
        cx = float(center_arr[0])
        cy = float(center_arr[1])

    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    translated = arr - np.array([cx, cy], dtype=float)
    rotation_matrix = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ],
        dtype=float,
    )
    rotated = translated @ rotation_matrix.T
    rotated += np.array([cx, cy], dtype=float)

    return [(float(x), float(y)) for x, y in rotated]


def generate_sine_curves(
    num_curves: int = 10,
    num_points: int = 200,
    amplitude: float = 1.0,
    offset_step: float = 0.1,
    phase_step: float = 0.0,
    x_start_min: float = 0.0,
    x_start_max: float = 10.0,
    rotation_step: float = 0.0,
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
        rotation_step: 曲線ごとの回転角度増分（度）
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

        if rotation_step != 0.0:
            curve = rotate_curve(curve, angle_degrees=i * rotation_step)

        curves.append(curve)

    return curves


def compute_all_pair_frechet_distances(
    curves: list[list[Point2D]],
) -> np.ndarray:
    """
    曲線群を総当たりで比較し、距離行列を返します。

    Returns:
        shape=(n, n) の距離行列
    """
    if not curves:
        raise ValueError("比較する曲線がありません。")

    n = len(curves)
    distances = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            dist = discrete_frechet_distance(curves[i], curves[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def cluster_curves_with_spectral_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    discrete Fréchet 距離行列を使ってスペクトラルクラスタリングします。

    Args:
        distance_matrix: shape=(n, n) の距離行列
        n_clusters: クラスタ数
        random_state: 乱数シード

    Returns:
        各曲線のクラスタラベル
    """
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix は正方行列である必要があります。")

    # 距離を類似度に変換する。
    # 小さい距離ほど大きい値になるよう、RBF風の変換を使う。
    positive_values = distance_matrix[distance_matrix > 0]
    scale = float(np.median(positive_values)) if positive_values.size > 0 else 1.0
    if scale <= 0:
        scale = 1.0

    affinity = np.exp(-(distance_matrix ** 2) / (2.0 * scale ** 2))
    np.fill_diagonal(affinity, 1.0)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
    )
    labels = model.fit_predict(affinity)
    return labels


def plot_curves(curves: list[list[Point2D]], labels: np.ndarray | None = None) -> None:
    """
    曲線群を matplotlib で表示します。

    labels が与えられた場合はクラスタごとに色分けします。
    """
    if not curves:
        raise ValueError("表示する曲線がありません。")

    fig, ax = plt.subplots(figsize=(12, 8))

    if labels is None:
        cmap = plt.get_cmap("viridis", len(curves))
        for i, curve in enumerate(curves):
            arr = np.asarray(curve, dtype=float)
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=cmap(i),
                linewidth=1.5,
                label=f"curve {i + 1}" if i < 10 else None,
            )
        if len(curves) <= 10:
            ax.legend(loc="best")
    else:
        unique_labels = sorted(set(int(x) for x in labels))
        cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
        for i, curve in enumerate(curves):
            label = int(labels[i])
            arr = np.asarray(curve, dtype=float)
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=cmap(label % 10),
                linewidth=1.8,
                label=f"cluster {label}" if f"cluster {label}" not in ax.get_legend_handles_labels()[1] else None,
            )
        ax.legend(loc="best")

    ax.set_title("Generated Sine Curves")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

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

    curves = generate_sine_curves(num_curves=10, seed=42, rotation_step=10.0)
    print(f"generated curves: {len(curves)}")
    print(f"first curve points: {len(curves[0])}")
    print(f"last curve points: {len(curves[-1])}")

    distance_matrix = compute_all_pair_frechet_distances(curves)
    print("pairwise distance matrix:")
    print(distance_matrix)

    labels = cluster_curves_with_spectral_clustering(distance_matrix, n_clusters=3)
    print("cluster labels:")
    for i, label in enumerate(labels):
        print(f"curve[{i}] -> cluster {label}")

    plot_curves(curves, labels=labels)


if __name__ == "__main__":
    main()
