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


def _rotate_array(arr: np.ndarray, angle_radians: float) -> np.ndarray:
    """
    numpy 配列の点列を原点周りに回転します。
    """
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    rotation_matrix = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ],
        dtype=float,
    )
    return arr @ rotation_matrix.T


def _normalize_by_covariance(
    arr: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    共分散行列を使って、曲線の向きを安定化させます。
    """
    if len(arr) < 2:
        return arr

    cov = np.cov(arr.T)
    if not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return arr

    eigvals, eigvecs = np.linalg.eigh(cov)

    # 主成分軸へ回転
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = math.atan2(float(principal[1]), float(principal[0]))
    arr = _rotate_array(arr, -angle)

    # 縦横のスケール差を少し緩和
    cov2 = np.cov(arr.T)
    if np.all(np.isfinite(cov2)) and cov2.shape == (2, 2):
        sx = math.sqrt(float(cov2[0, 0])) if cov2[0, 0] > eps else 1.0
        sy = math.sqrt(float(cov2[1, 1])) if cov2[1, 1] > eps else 1.0
        arr = arr / np.array([sx, sy], dtype=float)

    return arr


def normalize_curve(
    curve: Sequence[Sequence[float]],
    *,
    center: bool = True,
    align_first_point: bool = True,
    rotate: bool = True,
    scale: bool = True,
    unify_direction: bool = True,
    covariance_align: bool = True,
    eps: float = 1e-12,
) -> list[Point2D]:
    """
    曲線を正規化します。

    強化版の正規化内容:
    - center=True: 重心を原点へ移動
    - align_first_point=True: 最初の点を x 軸の正方向に揃える
    - rotate=True: PCA により主成分方向を x 軸へ揃える
    - covariance_align=True: 共分散に基づく追加整列
    - unify_direction=True: 方向の反転を抑制する
    - scale=True: 全体スケールを標準化する
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if center:
        centroid = arr.mean(axis=0)
        arr = arr - centroid

    if align_first_point and len(arr) >= 2:
        first_vec = arr[0]
        norm = float(np.linalg.norm(first_vec))
        if norm > eps:
            angle = math.atan2(float(first_vec[1]), float(first_vec[0]))
            arr = _rotate_array(arr, -angle)

    if rotate and len(arr) >= 2:
        cov = np.cov(arr.T)
        if np.all(np.isfinite(cov)) and cov.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, np.argmax(eigvals)]
            angle = math.atan2(float(principal[1]), float(principal[0]))
            arr = _rotate_array(arr, -angle)

    if covariance_align:
        arr = _normalize_by_covariance(arr, eps=eps)

    if unify_direction and len(arr) >= 2:
        start = arr[0]
        end = arr[-1]
        if (end[0] < start[0]) or (abs(end[0] - start[0]) < eps and end[1] < start[1]):
            arr = arr[::-1]

    if scale:
        scale_value = float(np.sqrt(np.mean(np.sum(arr ** 2, axis=1))))
        if scale_value > eps:
            arr = arr / scale_value

    if center:
        arr = arr - arr.mean(axis=0)

    return [(float(x), float(y)) for x, y in arr]


def discrete_frechet_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
) -> float:
    """
    2次元曲線同士の discrete Fréchet 距離を計算します。
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


def dtw_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
) -> float:
    """
    2次元曲線同士の DTW (Dynamic Time Warping) 距離を計算します。
    """
    p = _as_point_array(curve1)
    q = _as_point_array(curve2)

    n, m = len(p), len(q)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(p[i - 1] - q[j - 1]))
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    return float(dp[n, m])


def normalized_dtw_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
) -> float:
    """
    正規化前処理を施したうえで DTW 距離を計算します。
    """
    norm1 = normalize_curve(curve1)
    norm2 = normalize_curve(curve2)
    return dtw_distance(norm1, norm2)


def normalized_discrete_frechet_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
) -> float:
    """
    強化した正規化前処理を施したうえで discrete Fréchet 距離を計算します。
    """
    norm1 = normalize_curve(curve1)
    norm2 = normalize_curve(curve2)
    return discrete_frechet_distance(norm1, norm2)


def rotate_curve(
    curve: Sequence[Sequence[float]],
    angle_degrees: float,
    center: Sequence[float] | None = None,
) -> list[Point2D]:
    """
    曲線を指定角度だけ回転させます。
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
    translate_range: float = 15.0,
    seed: int | None = None,
) -> list[list[Point2D]]:
    """
    正弦波・矩形波・ノコギリ波をそれぞれ指定数ずつ生成します。
    種類のまとまりを弱めるため、各曲線にランダムな回転と平行移動を与えます。
    """
    if num_each_type < 1:
        raise ValueError("num_each_type は 1 以上で指定してください。")
    if num_points < 2:
        raise ValueError("num_points は 2 以上で指定してください。")
    if x_start_min > x_start_max:
        raise ValueError("x_start_min は x_start_max 以下で指定してください。")
    if sawtooth_period <= 0:
        raise ValueError("sawtooth_period は 0 より大きい値で指定してください。")
    if translate_range < 0:
        raise ValueError("translate_range は 0 以上で指定してください。")

    rng = np.random.default_rng(seed)
    curves: list[list[Point2D]] = []

    generators = (
        generate_sine_curve,
        generate_square_curve,
        generate_sawtooth_curve,
    )

    type_order = np.repeat(np.arange(3), num_each_type)
    rng.shuffle(type_order)

    for type_index in type_order:
        generator = generators[int(type_index)]

        x_start = float(rng.uniform(x_start_min, x_start_max))
        offset = float(rng.uniform(-offset_step * 3.0, offset_step * 3.0))
        phase = float(rng.uniform(-phase_step * math.pi, phase_step * math.pi))
        rotation = float(rng.uniform(-rotation_step, rotation_step))
        translate_dx = float(rng.uniform(-translate_range, translate_range))
        translate_dy = float(rng.uniform(-translate_range, translate_range))

        if generator is generate_sawtooth_curve:
            curve = generator(
                num_points=num_points,
                amplitude=amplitude,
                offset=offset,
                phase=phase,
                x_start=x_start,
                rotation_degrees=rotation,
                period=sawtooth_period,
                translate_dx=translate_dx,
                translate_dy=translate_dy,
            )
        else:
            curve = generator(
                num_points=num_points,
                amplitude=amplitude,
                offset=offset,
                phase=phase,
                x_start=x_start,
                rotation_degrees=rotation,
                translate_dx=translate_dx,
                translate_dy=translate_dy,
            )

        curves.append(curve)

    return curves


def compute_all_pair_distances(
    curves: list[list[Point2D]],
    metric: str = "frechet",
    normalize: bool = False,
) -> np.ndarray:
    """
    曲線群を総当たりで比較し、距離行列を返します。
    """
    if not curves:
        raise ValueError("比較する曲線がありません。")

    metric = metric.lower()
    if metric not in {"frechet", "dtw"}:
        raise ValueError("metric は 'frechet' または 'dtw' を指定してください。")

    n = len(curves)
    distances = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "frechet":
                dist = (
                    normalized_discrete_frechet_distance(curves[i], curves[j])
                    if normalize
                    else discrete_frechet_distance(curves[i], curves[j])
                )
            else:
                dist = (
                    normalized_dtw_distance(curves[i], curves[j])
                    if normalize
                    else dtw_distance(curves[i], curves[j])
                )
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


def find_distance_threshold_for_cluster_count(
    distance_matrix: np.ndarray,
    target_cluster_count: int,
    linkage_method: str = "average",
    candidate_count: int = 500,
) -> float:
    """
    目標クラスタ数になるしきい値を探索します。
    """
    if target_cluster_count < 1:
        raise ValueError("target_cluster_count は 1 以上で指定してください。")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix は正方行列である必要があります。")

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if len(Z) == 0:
        return 0.0

    merge_distances = np.unique(Z[:, 2])
    min_threshold = 0.0
    max_threshold = float(merge_distances[-1] + 1e-9)

    candidates = np.linspace(min_threshold, max_threshold, candidate_count)

    best_threshold = float(candidates[0])
    best_diff = float("inf")

    for threshold in candidates:
        labels = fcluster(Z, t=float(threshold), criterion="distance")
        cluster_count = len(set(int(x) for x in labels))
        diff = abs(cluster_count - target_cluster_count)

        if diff < best_diff:
            best_diff = diff
            best_threshold = float(threshold)

        if cluster_count == target_cluster_count:
            return float(threshold)

    return best_threshold


def plot_curves(
    curves: list[list[Point2D]],
    labels: np.ndarray | None = None,
    title: str = "Generated Curves",
) -> None:
    """
    曲線群を matplotlib で表示します。
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

    ax.set_title(title)
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
    title_prefix: str = "distance_threshold",
) -> None:
    """
    複数のしきい値で階層クラスタリングした結果をサブプロットで一括表示します。
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
        cluster_count = len(set(int(x) for x in labels))
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

        ax.set_title(f"{title_prefix} = {threshold:.4f} / clusters = {cluster_count}")
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

    frechet_dist = discrete_frechet_distance(curve_a, curve_b)
    dtw_dist = dtw_distance(curve_a, curve_b)
    print(f"discrete Fréchet distance: {frechet_dist:.6f}")
    print(f"DTW distance: {dtw_dist:.6f}")

    curves = generate_various_curves(
        num_each_type=3,
        seed=42,
        sawtooth_period=math.pi / 4.0,
        translate_range=20.0,
        rotation_step=45.0,
        offset_step=2.0,
        phase_step=1.0,
    )
    print(f"generated curves: {len(curves)}")
    print(f"first curve points: {len(curves[0])}")
    print(f"last curve points: {len(curves[-1])}")

    frechet_distance_matrix = compute_all_pair_distances(
        curves,
        metric="frechet",
        normalize=False,
    )
    dtw_distance_matrix = compute_all_pair_distances(
        curves,
        metric="dtw",
        normalize=False,
    )

    print("pairwise Fréchet distance matrix:")
    print(frechet_distance_matrix)
    print("pairwise DTW distance matrix:")
    print(dtw_distance_matrix)

    target_cluster_count = 3
    frechet_threshold = find_distance_threshold_for_cluster_count(
        frechet_distance_matrix,
        target_cluster_count=target_cluster_count,
        linkage_method="average",
    )
    dtw_threshold = find_distance_threshold_for_cluster_count(
        dtw_distance_matrix,
        target_cluster_count=target_cluster_count,
        linkage_method="average",
    )

    print(f"threshold for {target_cluster_count} clusters (Fréchet): {frechet_threshold:.6f}")
    print(f"threshold for {target_cluster_count} clusters (DTW): {dtw_threshold:.6f}")

    frechet_thresholds = np.linspace(
        max(0.0, frechet_threshold * 0.5),
        frechet_threshold * 1.5 if frechet_threshold > 0 else 1.0,
        4,
    ).tolist()
    dtw_thresholds = np.linspace(
        max(0.0, dtw_threshold * 0.5),
        dtw_threshold * 1.5 if dtw_threshold > 0 else 1.0,
        4,
    ).tolist()

    plot_clustering_threshold_sweep(
        curves,
        frechet_distance_matrix,
        thresholds=frechet_thresholds,
        linkage_method="average",
        title_prefix="Fréchet threshold",
    )
    plot_clustering_threshold_sweep(
        curves,
        dtw_distance_matrix,
        thresholds=dtw_thresholds,
        linkage_method="average",
        title_prefix="DTW threshold",
    )


if __name__ == "__main__":
    main()
