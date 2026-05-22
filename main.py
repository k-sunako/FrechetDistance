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


def normalize_curve(
    curve: Sequence[Sequence[float]],
    *,
    center: bool = True,
    scale: bool = True,
    eps: float = 1e-12,
) -> list[Point2D]:
    """
    Fourier記述子用に曲線を正規化します。

    - center=True: 重心を原点へ移動
    - scale=True: 全体スケールを標準化する
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if center:
        arr = arr - arr.mean(axis=0)

    if scale:
        scale_value = float(np.sqrt(np.mean(np.sum(arr ** 2, axis=1))))
        if scale_value > eps:
            arr = arr / scale_value

    return [(float(x), float(y)) for x, y in arr]


def fourier_descriptor(
    curve: Sequence[Sequence[float]],
    num_coefficients: int = 16,
    normalize: bool = True,
    use_magnitude_only: bool = False,
) -> np.ndarray:
    """
    曲線の Fourier 記述子を計算します。

    曲線を複素数列 z = x + i y として扱い、
    FFT の低周波成分を特徴量として取り出します。
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if normalize:
        arr = _as_point_array(normalize_curve(arr))

    if len(arr) < 2:
        return np.zeros(num_coefficients, dtype=float)

    z = arr[:, 0] + 1j * arr[:, 1]
    z = z - np.mean(z)

    spectrum = np.fft.fft(z)
    coeffs = spectrum[1 : num_coefficients + 1]

    if coeffs.size < num_coefficients:
        coeffs = np.pad(coeffs, (0, num_coefficients - coeffs.size), mode="constant")

    if use_magnitude_only:
        features = np.abs(coeffs).astype(float)
    else:
        if np.abs(coeffs[0]) > 1e-12:
            coeffs = coeffs / np.abs(coeffs[0])
        features = np.concatenate([coeffs.real, coeffs.imag]).astype(float)

    return features


def fourier_descriptor_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
    num_coefficients: int = 16,
    use_magnitude_only: bool = False,
) -> float:
    """
    Fourier 記述子同士のユークリッド距離を計算します。
    """
    d1 = fourier_descriptor(
        curve1,
        num_coefficients=num_coefficients,
        normalize=True,
        use_magnitude_only=use_magnitude_only,
    )
    d2 = fourier_descriptor(
        curve2,
        num_coefficients=num_coefficients,
        normalize=True,
        use_magnitude_only=use_magnitude_only,
    )
    return float(np.linalg.norm(d1 - d2))


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
    num_coefficients: int = 16,
    use_magnitude_only: bool = False,
) -> np.ndarray:
    """
    Fourier 記述子による全ペア距離行列を作成します。
    """
    if not curves:
        raise ValueError("比較する曲線がありません。")

    n = len(curves)
    distances = np.zeros((n, n), dtype=float)

    descriptors = [
        fourier_descriptor(
            curve,
            num_coefficients=num_coefficients,
            normalize=True,
            use_magnitude_only=use_magnitude_only,
        )
        for curve in curves
    ]

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(descriptors[i] - descriptors[j]))
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
        seen = set()
        for i, curve in enumerate(curves):
            label = int(labels[i])
            arr = np.asarray(curve, dtype=float)
            legend_label = f"cluster {label}" if label not in seen else None
            seen.add(label)
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=cmap((label - 1) % 10),
                linewidth=1.8,
                label=legend_label,
            )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_curves_side_by_side(
    curves: list[list[Point2D]],
    labels: np.ndarray | None = None,
    title_left: str = "Original Curves",
    title_right: str = "Fourier Descriptor Comparison",
) -> None:
    """
    左右2枚のサブプロットで曲線群を比較表示します。
    左は元の曲線、右はラベル付きの表示です。
    """
    if not curves:
        raise ValueError("表示する曲線がありません。")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_left, ax_right = axes

    cmap_left = plt.get_cmap("viridis", len(curves))
    for i, curve in enumerate(curves):
        arr = np.asarray(curve, dtype=float)
        ax_left.plot(arr[:, 0], arr[:, 1], color=cmap_left(i), linewidth=1.5)
    ax_left.set_title(title_left)
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.grid(True, alpha=0.3)

    if labels is None:
        cmap_right = plt.get_cmap("viridis", len(curves))
        for i, curve in enumerate(curves):
            arr = np.asarray(curve, dtype=float)
            ax_right.plot(arr[:, 0], arr[:, 1], color=cmap_right(i), linewidth=1.5)
    else:
        unique_labels = sorted(set(int(x) for x in labels))
        cmap_right = plt.get_cmap("tab10", max(len(unique_labels), 1))
        seen = set()
        for i, curve in enumerate(curves):
            label = int(labels[i])
            arr = np.asarray(curve, dtype=float)
            legend_label = f"cluster {label}" if label not in seen else None
            seen.add(label)
            ax_right.plot(
                arr[:, 0],
                arr[:, 1],
                color=cmap_right((label - 1) % 10),
                linewidth=1.8,
                label=legend_label,
            )
        ax_right.legend(loc="best")

    ax_right.set_title(title_right)
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(
    curves: list[list[Point2D]],
    num_coefficients: int = 16,
    use_magnitude_only: bool = False,
    target_cluster_count: int = 3,
    linkage_method: str = "average",
) -> None:
    """
    Fourier記述子ベースの比較をサブプロットで表示します。
    """
    if not curves:
        raise ValueError("表示する曲線がありません。")

    distance_matrix = compute_all_pair_distances(
        curves,
        num_coefficients=num_coefficients,
        use_magnitude_only=use_magnitude_only,
    )
    threshold = find_distance_threshold_for_cluster_count(
        distance_matrix,
        target_cluster_count=target_cluster_count,
        linkage_method=linkage_method,
    )
    labels = cluster_curves_with_hierarchical_clustering(
        distance_matrix,
        distance_threshold=threshold,
        linkage_method=linkage_method,
    )

    plot_curves_side_by_side(
        curves,
        labels=labels,
        title_left="Original Curves",
        title_right=f"Fourier Descriptor Clustering / threshold={threshold:.4f}",
    )


def main() -> None:
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

    distance_matrix = compute_all_pair_distances(
        curves,
        num_coefficients=16,
        use_magnitude_only=False,
    )
    print("pairwise Fourier descriptor distance matrix:")
    print(distance_matrix)

    target_cluster_count = 3
    threshold = find_distance_threshold_for_cluster_count(
        distance_matrix,
        target_cluster_count=target_cluster_count,
        linkage_method="average",
    )
    print(f"threshold for {target_cluster_count} clusters: {threshold:.6f}")

    labels = cluster_curves_with_hierarchical_clustering(
        distance_matrix,
        distance_threshold=threshold,
        linkage_method="average",
    )
    print("cluster labels:")
    for i, label in enumerate(labels):
        print(f"curve[{i}] -> cluster {label}")

    plot_curves_side_by_side(
        curves,
        labels=labels,
        title_left="Generated Curves",
        title_right="Fourier Descriptor Clustering",
    )
    plot_metric_comparison(
        curves,
        num_coefficients=16,
        use_magnitude_only=False,
        target_cluster_count=3,
        linkage_method="average",
    )


if __name__ == "__main__":
    main()
