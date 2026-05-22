from __future__ import annotations

from typing import Sequence, Any

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid


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
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if center:
        arr = arr - arr.mean(axis=0)

    if scale:
        scale_value = float(np.sqrt(np.mean(np.sum(arr ** 2, axis=1))))
        if scale_value > eps:
            arr = arr / scale_value

    return [(float(x), float(y)) for x, y in arr]


def _resample_curve(
    curve: Sequence[Sequence[float]],
    num_points: int,
) -> np.ndarray:
    """
    曲線を弧長パラメータで等間隔に再サンプリングします。
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if num_points < 2:
        raise ValueError("num_points は 2 以上で指定してください。")

    diffs = np.diff(arr, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    total_length = float(np.sum(segment_lengths))

    if total_length <= 1e-12:
        return np.repeat(arr[:1], num_points, axis=0)

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target = np.linspace(0.0, total_length, num_points)

    x = np.interp(target, cumulative, arr[:, 0])
    y = np.interp(target, cumulative, arr[:, 1])
    return np.column_stack([x, y])


def _align_start_point(curve: np.ndarray) -> np.ndarray:
    """
    曲線の開始点依存を弱めるため、最も左下に近い点を開始点に回転します。
    """
    if len(curve) == 0:
        return curve

    order = np.lexsort((curve[:, 1], curve[:, 0]))
    start_index = int(order[0])
    return np.roll(curve, -start_index, axis=0)


def _pca_align_curve(curve: np.ndarray) -> np.ndarray:
    """
    主成分軸に揃えて回転不変性を高めます。
    """
    if len(curve) < 2:
        return curve

    centered = curve - curve.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle = math.atan2(float(principal_axis[1]), float(principal_axis[0]))

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rotation_matrix = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ],
        dtype=float,
    )
    aligned = centered @ rotation_matrix.T

    if np.mean(aligned[:, 1]) < 0:
        aligned[:, 1] *= -1.0

    return aligned


def _standardize_curve_for_descriptor(
    curve: Sequence[Sequence[float]],
    *,
    num_points: int | None = None,
    use_pca_alignment: bool = True,
    use_start_alignment: bool = True,
    center: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """
    Fourier記述子用の前処理をまとめて行います。
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if num_points is not None:
        arr = _resample_curve(arr, num_points=num_points)

    if center:
        arr = arr - arr.mean(axis=0)

    if scale:
        scale_value = float(np.sqrt(np.mean(np.sum(arr ** 2, axis=1))))
        if scale_value > 1e-12:
            arr = arr / scale_value

    if use_pca_alignment:
        arr = _pca_align_curve(arr)

    if use_start_alignment:
        arr = _align_start_point(arr)

    return arr


def fourier_descriptor(
    curve: Sequence[Sequence[float]],
    num_coefficients: int = 16,
    normalize: bool = True,
    use_magnitude_only: bool = False,
    resample_points: int | None = 256,
    use_pca_alignment: bool = True,
    use_start_alignment: bool = True,
) -> np.ndarray:
    """
    曲線の Fourier 記述子を計算します。
    """
    arr = _as_point_array(curve).astype(float, copy=True)

    if normalize:
        arr = _standardize_curve_for_descriptor(
            arr,
            num_points=resample_points,
            use_pca_alignment=use_pca_alignment,
            use_start_alignment=use_start_alignment,
            center=True,
            scale=True,
        )
    elif resample_points is not None:
        arr = _resample_curve(arr, num_points=resample_points)

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
        base = coeffs[0]
        if np.abs(base) > 1e-12:
            coeffs = coeffs / base
        features = np.concatenate([coeffs.real, coeffs.imag]).astype(float)

    return features


def fourier_descriptor_distance(
    curve1: Sequence[Sequence[float]],
    curve2: Sequence[Sequence[float]],
    num_coefficients: int = 16,
    use_magnitude_only: bool = False,
    resample_points: int | None = 256,
    use_pca_alignment: bool = True,
    use_start_alignment: bool = True,
) -> float:
    """
    Fourier 記述子同士のユークリッド距離を計算します。
    """
    d1 = fourier_descriptor(
        curve1,
        num_coefficients=num_coefficients,
        normalize=True,
        use_magnitude_only=use_magnitude_only,
        resample_points=resample_points,
        use_pca_alignment=use_pca_alignment,
        use_start_alignment=use_start_alignment,
    )
    d2 = fourier_descriptor(
        curve2,
        num_coefficients=num_coefficients,
        normalize=True,
        use_magnitude_only=use_magnitude_only,
        resample_points=resample_points,
        use_pca_alignment=use_pca_alignment,
        use_start_alignment=use_start_alignment,
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


def generate_various_curves(
    num_each_type: int = 3,
    num_points: int = 200,
    amplitude: float = 1.0,
    offset_step: float = 0.3,
    phase_step: float = 0.4,
    x_start_min: float = 0.0,
    x_start_max: float = 10.0,
    rotation_step: float = 10.0,
    translate_range: float = 15.0,
    seed: int | None = None,
) -> list[list[Point2D]]:
    """
    正弦波・矩形波をそれぞれ指定数ずつ生成します。
    種類のまとまりを弱めるため、各曲線にランダムな回転と平行移動を与えます。
    """
    if num_each_type < 1:
        raise ValueError("num_each_type は 1 以上で指定してください。")
    if num_points < 2:
        raise ValueError("num_points は 2 以上で指定してください。")
    if x_start_min > x_start_max:
        raise ValueError("x_start_min は x_start_max 以下で指定してください。")
    if translate_range < 0:
        raise ValueError("translate_range は 0 以上で指定してください。")

    rng = np.random.default_rng(seed)
    curves: list[list[Point2D]] = []

    generators = (
        generate_sine_curve,
        generate_square_curve,
    )

    type_order = np.repeat(np.arange(len(generators)), num_each_type)
    rng.shuffle(type_order)

    for type_index in type_order:
        generator = generators[int(type_index)]

        x_start = float(rng.uniform(x_start_min, x_start_max))
        offset = float(rng.uniform(-offset_step * 3.0, offset_step * 3.0))
        phase = float(rng.uniform(-phase_step * math.pi, phase_step * math.pi))
        rotation = float(rng.uniform(-rotation_step, rotation_step))
        translate_dx = float(rng.uniform(-translate_range, translate_range))
        translate_dy = float(rng.uniform(-translate_range, translate_range))

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
    resample_points: int | None = 256,
    use_pca_alignment: bool = True,
    use_start_alignment: bool = True,
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
            resample_points=resample_points,
            use_pca_alignment=use_pca_alignment,
            use_start_alignment=use_start_alignment,
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


def evaluate_clustering(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    クラスタリング結果を評価する指標を計算します。
    """
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix は正方行列である必要があります。")
    if len(labels) != distance_matrix.shape[0]:
        raise ValueError("labels の長さが distance_matrix と一致していません。")

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }

    try:
        silhouette = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
    except Exception:
        silhouette = float("nan")

    try:
        ch = float(calinski_harabasz_score(distance_matrix, labels))
    except Exception:
        ch = float("nan")

    try:
        db = float(davies_bouldin_score(distance_matrix, labels))
    except Exception:
        db = float("nan")

    return {
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }


def score_clustering_metrics(metrics: dict[str, float]) -> float:
    """
    最適化用のスコアに変換します。
    silhouette と CH は高いほど良い、DB は低いほど良い。
    """
    score = 0.0

    silhouette = metrics.get("silhouette", float("nan"))
    ch = metrics.get("calinski_harabasz", float("nan"))
    db = metrics.get("davies_bouldin", float("nan"))

    if np.isfinite(silhouette):
        score += silhouette * 2.0
    if np.isfinite(ch):
        score += math.log1p(max(ch, 0.0)) * 0.1
    if np.isfinite(db):
        score -= db * 0.5

    return float(score)


def optimize_clustering_parameters(
    curves: list[list[Point2D]],
    param_grid: dict[str, list[Any]],
    target_cluster_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    scikit-learn を使ってパラメータ候補を総当たりし、最適な組み合わせを探索します。
    """
    if not curves:
        raise ValueError("比較する曲線がありません。")
    if not param_grid:
        raise ValueError("param_grid が空です。")

    results: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = -float("inf")

    for params in ParameterGrid(param_grid):
        distance_matrix = compute_all_pair_distances(
            curves,
            num_coefficients=int(params.get("num_coefficients", 16)),
            use_magnitude_only=bool(params.get("use_magnitude_only", False)),
            resample_points=params.get("resample_points", 256),
            use_pca_alignment=bool(params.get("use_pca_alignment", True)),
            use_start_alignment=bool(params.get("use_start_alignment", True)),
        )

        if "distance_threshold" in params and params["distance_threshold"] is not None:
            labels = cluster_curves_with_hierarchical_clustering(
                distance_matrix,
                distance_threshold=float(params["distance_threshold"]),
                linkage_method=str(params.get("linkage_method", "average")),
            )
        else:
            n_clusters = params.get("n_clusters", target_cluster_count)
            if n_clusters is None:
                raise ValueError("n_clusters か distance_threshold を指定してください。")
            labels = cluster_curves_with_hierarchical_clustering(
                distance_matrix,
                n_clusters=int(n_clusters),
                linkage_method=str(params.get("linkage_method", "average")),
            )

        metrics = evaluate_clustering(distance_matrix, labels)
        score = score_clustering_metrics(metrics)

        if target_cluster_count is not None:
            cluster_count = len(np.unique(labels))
            score -= abs(cluster_count - target_cluster_count) * 10.0

        row = {
            "params": dict(params),
            "score": score,
            "labels": labels,
            "metrics": metrics,
        }
        results.append(row)

        if score > best_score:
            best_score = score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("最適化に失敗しました。")

    results.sort(key=lambda x: x["score"], reverse=True)
    return best_params, results


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
        num_each_type=6,
        seed=42,
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
        resample_points=256,
        use_pca_alignment=True,
        use_start_alignment=True,
    )
    print("pairwise Fourier descriptor distance matrix:")
    print(distance_matrix)

    target_cluster_count = 2
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

    print("clustering metrics:")
    metrics = evaluate_clustering(distance_matrix, labels)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    param_grid = {
        "num_coefficients": [4, 8, 16, 32],
        "use_magnitude_only": [False, True],
        "resample_points": [64, 128, 256],
        "use_pca_alignment": [False, True],
        "use_start_alignment": [False, True],
        "linkage_method": ["average", "complete", "single"],
    }

    best_params, results = optimize_clustering_parameters(
        curves,
        param_grid=param_grid,
        target_cluster_count=2,
    )
    print("best params:")
    print(best_params)
    print("top 5 results:")
    for row in results[:5]:
        print(row["score"], row["params"], row["metrics"])

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
        target_cluster_count=2,
        linkage_method="average",
    )


if __name__ == "__main__":
    main()
