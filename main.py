from __future__ import annotations

from typing import Sequence

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


Point2D = tuple[float, float]


def _as_point_array(curve: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("曲線は shape=(N, 2) の2次元座標列で指定してください。")
    if arr.shape[0] < 1:
        raise ValueError("曲線は少なくとも1点以上必要です。")
    return arr


def translate_curve(
    curve: Sequence[Sequence[float]],
    dx: float = 0.0,
    dy: float = 0.0,
) -> list[Point2D]:
    """
    曲線を指定量だけ平行移動します。
    """
    arr = _as_point_array(curve)
    translated = arr + np.array([dx, dy], dtype=float)
    return [(float(x), float(y)) for x, y in translated]


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


def generate_sine_curve(
    num_points: int = 200,
    amplitude: float = 1.0,
    offset: float = 0.0,
    phase: float = 0.0,
    x_start: float = 0.0,
    rotation_degrees: float = 0.0,
    translate_dx: float = 0.0,
    translate_dy: float = 0.0,
) -> list[Point2D]:
    """
    正弦波曲線を1本生成します。
    """
    x_end = x_start + 2.0 * math.pi
    x_values = np.linspace(x_start, x_end, num_points)
    curve = [
        (float(x), float(amplitude * math.sin(x + phase) + offset))
        for x in x_values
    ]

    if rotation_degrees != 0.0:
        curve = rotate_curve(curve, angle_degrees=rotation_degrees)

    if translate_dx != 0.0 or translate_dy != 0.0:
        curve = translate_curve(curve, dx=translate_dx, dy=translate_dy)

    return curve


def generate_square_curve(
    num_points: int = 200,
    amplitude: float = 1.0,
    offset: float = 0.0,
    phase: float = 0.0,
    x_start: float = 0.0,
    rotation_degrees: float = 0.0,
    translate_dx: float = 0.0,
    translate_dy: float = 0.0,
) -> list[Point2D]:
    """
    矩形波曲線を1本生成します。
    """
    x_end = x_start + 2.0 * math.pi
    x_values = np.linspace(x_start, x_end, num_points)
    curve = [
        (float(x), float(amplitude * np.sign(math.sin(x + phase)) + offset))
        for x in x_values
    ]

    if rotation_degrees != 0.0:
        curve = rotate_curve(curve, angle_degrees=rotation_degrees)

    if translate_dx != 0.0 or translate_dy != 0.0:
        curve = translate_curve(curve, dx=translate_dx, dy=translate_dy)

    return curve


def generate_sawtooth_curve(
    num_points: int = 200,
    amplitude: float = 1.0,
    offset: float = 0.0,
    phase: float = 0.0,
    x_start: float = 0.0,
    rotation_degrees: float = 0.0,
    period: float = 2.0 * math.pi,
    translate_dx: float = 0.0,
    translate_dy: float = 0.0,
) -> list[Point2D]:
    """
    ノコギリ波曲線を1本生成します。
    """
    x_end = x_start + 2.0 * math.pi
    x_values = np.linspace(x_start, x_end, num_points)

    shifted = x_values + phase
    saw = 2.0 * ((shifted / period) - np.floor(0.5 + shifted / period))
    curve = [
        (float(x), float(amplitude * y + offset))
        for x, y in zip(x_values, saw, strict=True)
    ]

    if rotation_degrees != 0.0:
        curve = rotate_curve(curve, angle_degrees=rotation_degrees)

    if translate_dx != 0.0 or translate_dy != 0.0:
        curve = translate_curve(curve, dx=translate_dx, dy=translate_dy)

    return curve


def generate_various_curves(
    num_each_type: int = 3,
    num_points: int = 200,
    amplitude: float = 1.0,
    offset_step: float = 0.3,
    phase_step: float = 0.4,
    x_start_min: float = 0.0,
    x_start_max: float = 10.0,
    rotation_step: float = 10.0,
    sawtooth_period: float = math.pi,
    type_translation_dx: float = 8.0,
    type_translation_dy: float = 6.0,
    intra_type_translation_dx: float = 1.2,
    intra_type_translation_dy: float = 0.8,
    seed: int | None = None,
) -> list[list[Point2D]]:
    """
    正弦波・矩形波・ノコギリ波をそれぞれ指定数ずつ生成します。
    各タイプ内の3本は、同一の元曲線を回転角と平行移動で変化させます。
    """
    if num_each_type < 1:
        raise ValueError("num_each_type は 1 以上で指定してください。")
    if num_points < 2:
        raise ValueError("num_points は 2 以上で指定してください。")
    if x_start_min > x_start_max:
        raise ValueError("x_start_min は x_start_max 以下で指定してください。")
    if sawtooth_period <= 0:
        raise ValueError("sawtooth_period は 0 より大きい値で指定してください。")

    rng = np.random.default_rng(seed)
    curves: list[list[Point2D]] = []

    x_start_by_type = [
        float(rng.uniform(x_start_min, x_start_max)),
        float(rng.uniform(x_start_min, x_start_max)),
        float(rng.uniform(x_start_min, x_start_max)),
    ]

    type_offsets = [
        (0.0 * type_translation_dx, 0.0 * type_translation_dy),
        (1.0 * type_translation_dx, 1.0 * type_translation_dy),
        (2.0 * type_translation_dx, 2.0 * type_translation_dy),
    ]

    for type_index in range(3):
        x_start = x_start_by_type[type_index]
        offset = type_index * offset_step
        phase = type_index * phase_step
        base_dx, base_dy = type_offsets[type_index]

        for i in range(num_each_type):
            rotation = i * rotation_step
            intra_dx = i * intra_type_translation_dx
            intra_dy = i * intra_type_translation_dy

            if type_index == 0:
                curve = generate_sine_curve(
                    num_points=num_points,
                    amplitude=amplitude,
                    offset=offset,
                    phase=phase,
                    x_start=x_start,
                    rotation_degrees=rotation,
                    translate_dx=base_dx + intra_dx,
                    translate_dy=base_dy + intra_dy,
                )
            elif type_index == 1:
                curve = generate_square_curve(
                    num_points=num_points,
                    amplitude=amplitude,
                    offset=offset,
                    phase=phase,
                    x_start=x_start,
                    rotation_degrees=rotation,
                    translate_dx=base_dx + intra_dx,
                    translate_dy=base_dy + intra_dy,
                )
            else:
                curve = generate_sawtooth_curve(
                    num_points=num_points,
                    amplitude=amplitude,
                    offset=offset,
                    phase=phase,
                    x_start=x_start,
                    rotation_degrees=rotation,
                    period=sawtooth_period,
                    translate_dx=base_dx + intra_dx,
                    translate_dy=base_dy + intra_dy,
                )

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


def cluster_curves_with_hierarchical_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    linkage_method: str = "average",
) -> np.ndarray:
    """
    距離行列を使って階層クラスタリングします。

    Args:
        distance_matrix: shape=(n, n) の距離行列
        n_clusters: クラスタ数を指定する場合
        distance_threshold: 距離しきい値で分割する場合
        linkage_method: linkage の手法 ("average", "complete", "single" など)

    Returns:
        各曲線のクラスタラベル
    """
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix は正方行列である必要があります。")

    if n_clusters is None and distance_threshold is None:
        raise ValueError("n_clusters か distance_threshold のどちらかを指定してください。")

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

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
                color=cmap((label - 1) % 10),
                linewidth=1.8,
                label=f"cluster {label}" if f"cluster {label}" not in ax.get_legend_handles_labels()[1] else None,
            )
        ax.legend(loc="best")

    ax.set_title("Generated Curves")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_clustering_threshold_sweep(
    curves: list[list[Point2D]],
    distance_matrix: np.ndarray,
    thresholds: Sequence[float],
    linkage_method: str = "average",
) -> None:
    """
    複数のしきい値で階層クラスタリングした結果をサブプロットで一括表示します。
    各サブプロットのタイトルにクラスタ数も表示します。
    """
    if not curves:
        raise ValueError("表示する曲線がありません。")
    if len(thresholds) < 1:
        raise ValueError("thresholds は1つ以上指定してください。")

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)

    n_plots = len(thresholds)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 6 * n_rows),
        squeeze=False,
    )

    for idx, threshold in enumerate(thresholds):
        ax = axes[idx // n_cols][idx % n_cols]
        labels = fcluster(Z, t=threshold, criterion="distance")
        unique_labels = sorted(set(int(x) for x in labels))
        cluster_count = len(unique_labels)
        cmap = plt.get_cmap("tab10", max(cluster_count, 1))

        for i, curve in enumerate(curves):
            label = int(labels[i])
            arr = np.asarray(curve, dtype=float)
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=cmap((label - 1) % 10),
                linewidth=1.5,
            )

        ax.set_title(f"distance_threshold = {threshold} / clusters = {cluster_count}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    plt.tight_layout()
    plt.show()


def main() -> None:
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

    curves = generate_various_curves(num_each_type=3, seed=42, sawtooth_period=math.pi / 2.0)
    print(f"generated curves: {len(curves)}")
    print(f"first curve points: {len(curves[0])}")
    print(f"last curve points: {len(curves[-1])}")

    distance_matrix = compute_all_pair_frechet_distances(curves)
    print("pairwise distance matrix:")
    print(distance_matrix)

    thresholds = np.linspace(0.5, 10.0, 4).tolist()
    plot_clustering_threshold_sweep(
        curves,
        distance_matrix,
        thresholds=thresholds,
        linkage_method="average",
    )


if __name__ == "__main__":
    main()
