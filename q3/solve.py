import time
from pathlib import Path
from typing import Any as Q3Any
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np

q3_started = time.perf_counter()

DATA = Path("data/q3")
OUT = Path("outputs/q3")


def load_images():
    paths = [DATA / f"Q3_Image_{index}.jpg" for index in range(1, 7)]
    images = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in images):
        raise FileNotFoundError(f"Could not load all six panorama inputs from {DATA}.")
    return images


def border_black_fraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black = (gray <= 8).astype(np.uint8)
    count, labels = cv2.connectedComponents(black, connectivity=8)
    if count <= 1:
        return 0.0
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    return float(np.isin(labels, border_labels).sum() / labels.size)


def order_corners(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered.sum(axis=1))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered


def cross2d_q3(first, second):
    return first[0] * second[1] - first[1] * second[0]


def fit_quad_sides(contour, quad):
    points = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        direction = end - start
        length = np.linalg.norm(direction)
        unit = direction / max(length, 1e-6)
        relative = points - start
        projection = relative @ unit
        distance = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
        selected = points[
            (projection >= -3) & (projection <= length + 3) & (distance <= 3)
        ]
        if len(selected) < 8:
            return quad
        fitted = cv2.fitLine(
            selected.reshape(-1, 1, 2), cv2.DIST_WELSCH, 0, 0.01, 0.01
        ).ravel()
        lines.append(
            (
                np.array([fitted[2], fitted[3]], dtype=np.float32),
                np.array([fitted[0], fitted[1]], dtype=np.float32),
            )
        )
    corners = []
    for index in range(4):
        previous_point, previous_direction = lines[(index - 1) % 4]
        point, direction = lines[index]
        denominator = cross2d_q3(previous_direction, direction)
        if abs(denominator) < 1e-5:
            return quad
        scale = cross2d_q3(point - previous_point, direction) / denominator
        corners.append(previous_point + scale * previous_direction)
    corners = np.asarray(corners, dtype=np.float32)
    if not cv2.isContourConvex(np.round(corners).astype(np.int32)):
        return quad
    if max(np.ptp(corners[:, 0]), np.ptp(corners[:, 1])) > 1.5 * max(
        np.ptp(quad[:, 0]), np.ptp(quad[:, 1])
    ):
        return quad
    return corners


def find_valid_quad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = (gray > 8).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No valid-content contour found in distorted image.")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    quad = None
    for fraction in np.arange(0.004, 0.081, 0.002):
        approximation = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if len(approximation) == 4:
            quad = approximation[:, 0, :].astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(contour))
    quad = order_corners(quad)
    h, w = image.shape[:2]
    clipped = (
        np.any(quad[:, 0] <= 1)
        or np.any(quad[:, 0] >= w - 2)
        or np.any(quad[:, 1] <= 1)
        or np.any(quad[:, 1] >= h - 2)
    )
    if clipped:
        quad = fit_quad_sides(contour, quad)
    return quad


def correct_distortions(images):
    corrected = []
    masks = []
    records = []
    for index, image in enumerate(images):
        fraction = border_black_fraction(image)
        is_distorted = fraction > 0.015
        h, w = image.shape[:2]
        valid_mask = np.ones((h, w), dtype=np.uint8) * 255
        if is_distorted:
            quad = find_valid_quad(image)
            destination = np.asarray(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(quad, destination)
            warped = cv2.warpPerspective(
                image,
                transform,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            quad = None
            transform = np.eye(3, dtype=np.float64)
            warped = image.copy()
        corrected.append(warped)
        masks.append(valid_mask)
        records.append(
            {
                "index": index,
                "black_fraction": fraction,
                "distorted": is_distorted,
                "quad": quad,
                "transform": transform,
            }
        )
    return corrected, masks, records


def extract_features(images):
    sift = cast(Q3Any, cv2).SIFT_create(nfeatures=5000)
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(sift.detectAndCompute(gray, None))
    return features


def estimate_pair(feature_i, feature_j):
    keypoints_i, descriptors_i = feature_i
    keypoints_j, descriptors_j = feature_j
    if descriptors_i is None or descriptors_j is None:
        return None, 0, 0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_i, descriptors_j, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    if len(good) < 4:
        return None, 0, 0
    points_i = np.asarray([keypoints_i[m.queryIdx].pt for m in good], dtype=np.float32)
    points_j = np.asarray([keypoints_j[m.trainIdx].pt for m in good], dtype=np.float32)
    homography, inlier_mask = cv2.findHomography(
        points_i, points_j, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999
    )
    if homography is None or inlier_mask is None:
        return None, 0, 0
    return homography, int(inlier_mask.sum()), len(good)


def pairwise_homographies(features):
    count = len(features)
    homographies = {}
    inliers = np.zeros((count, count), dtype=np.int32)
    ratios = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            homography, n_inliers, n_good = estimate_pair(features[i], features[j])
            if homography is None:
                continue
            homographies[(i, j)] = homography
            inliers[i, j] = inliers[j, i] = n_inliers
            ratios[i, j] = ratios[j, i] = n_inliers / max(n_good, 1)
    return homographies, inliers, ratios


def cylindrical_warp(image, mask, focal_length=640.0):
    h, w = image.shape[:2]
    center_x, center_y = (w - 1) / 2.0, (h - 1) / 2.0
    destination_x, destination_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    angle = (destination_x - center_x) / focal_length
    source_x = focal_length * np.tan(angle) + center_x
    source_y = (destination_y - center_y) * np.sqrt(
        (source_x - center_x) ** 2 + focal_length**2
    ) / focal_length + center_y
    map_x = source_x.astype(np.float32)
    map_y = source_y.astype(np.float32)
    warped_image = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_mask = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    return warped_image, warped_mask


def pairwise_cylindrical_translations(features):
    count = len(features)
    translations = {}
    inliers = np.zeros((count, count), dtype=np.int32)
    ratios = np.zeros((count, count), dtype=np.float32)
    errors = {}
    for i in range(count):
        for j in range(i + 1, count):
            keypoints_i, descriptors_i = features[i]
            keypoints_j, descriptors_j = features[j]
            if descriptors_i is None or descriptors_j is None:
                errors[(i, j)] = np.empty(0, dtype=np.float32)
                continue
            matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
                descriptors_i, descriptors_j, k=2
            )
            good = [a for a, b in matches if a.distance < 0.75 * b.distance]
            if len(good) < 4:
                errors[(i, j)] = np.empty(0, dtype=np.float32)
                continue
            points_i = np.asarray(
                [keypoints_i[m.queryIdx].pt for m in good], dtype=np.float32
            )
            points_j = np.asarray(
                [keypoints_j[m.trainIdx].pt for m in good], dtype=np.float32
            )
            homography, inlier_mask = cv2.findHomography(
                points_i,
                points_j,
                cv2.USAC_MAGSAC,
                3.0,
                maxIters=10000,
                confidence=0.999,
            )
            if homography is None or inlier_mask is None:
                errors[(i, j)] = np.empty(0, dtype=np.float32)
                continue
            inlier_mask = inlier_mask.ravel().astype(bool)
            projected = cv2.perspectiveTransform(
                points_i[inlier_mask, None, :], homography
            )[:, 0]
            pair_errors = np.linalg.norm(projected - points_j[inlier_mask], axis=1)
            delta = np.median(points_j[inlier_mask] - points_i[inlier_mask], axis=0)
            translations[(i, j)] = np.array(
                [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            inliers[i, j] = inliers[j, i] = int(inlier_mask.sum())
            ratio = inlier_mask.sum() / max(len(good), 1)
            ratios[i, j] = ratios[j, i] = ratio
            errors[(i, j)] = pair_errors
    return translations, inliers, ratios, errors


def maximum_spanning_tree(homographies, inliers, ratios):
    count = inliers.shape[0]
    accepted = {
        edge
        for edge, homography in homographies.items()
        if inliers[edge] >= 30 and ratios[edge] >= 0.20
    }

    def grow_tree(edges):
        tree = []
        included = {0}
        while len(included) < count:
            options = []
            for i, j in edges:
                if (i in included) != (j in included):
                    options.append((int(inliers[i, j]), i, j))
            if not options:
                return None
            weight, i, j = max(options)
            tree.append((i, j, weight))
            included.add(j if i in included else i)
        return tree

    tree = grow_tree(accepted)
    if tree is None:
        fallback = {
            edge for edge in homographies if inliers[edge] >= 8 and ratios[edge] >= 0.10
        }
        tree = grow_tree(fallback)
    if tree is None:
        raise RuntimeError(
            "The pairwise overlap graph is disconnected; cannot build a panorama tree."
        )
    return tree


def tree_maps_to_root(tree, homographies, count, root):
    adjacency = {index: {} for index in range(count)}
    for i, j, _ in tree:
        h_i_j = homographies[(min(i, j), max(i, j))]
        if i > j:
            h_i_j = np.linalg.inv(h_i_j)
        adjacency[i][j] = np.linalg.inv(h_i_j)
        adjacency[j][i] = h_i_j
    transforms = {root: np.eye(3, dtype=np.float64)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child, h_child_parent in adjacency[parent].items():
            if child in transforms:
                continue
            transforms[child] = transforms[parent] @ h_child_parent
            stack.append(child)
    return transforms


def mapped_center(transform, shape):
    h, w = shape[:2]
    point = np.asarray([[[(w - 1) / 2, (h - 1) / 2]]], dtype=np.float32)
    return cv2.perspectiveTransform(point, transform)[0, 0]


def projectivity_measure(homography, image_shape):
    matrix = homography / homography[2, 2]
    return float(np.linalg.norm(matrix[2, :2]) * np.hypot(*image_shape[:2]))


def stitch(corrected, masks, homographies, inliers, ratios, tree):
    count = len(corrected)
    root_maps = tree_maps_to_root(tree, homographies, count, root=0)
    positions = {
        index: mapped_center(root_maps[index], corrected[index].shape)
        for index in range(count)
    }
    order = sorted(range(count), key=lambda index: positions[index][0])
    reference = order[len(order) // 2]
    root_to_reference = np.linalg.inv(root_maps[reference])
    image_to_reference = {
        index: root_to_reference @ root_maps[index] for index in range(count)
    }

    corners = np.asarray(
        [
            [0, 0],
            [corrected[0].shape[1], 0],
            [corrected[0].shape[1], corrected[0].shape[0]],
            [0, corrected[0].shape[0]],
        ],
        dtype=np.float32,
    )
    all_corners = [
        cv2.perspectiveTransform(corners[None, :, :], image_to_reference[index])[0]
        for index in range(count)
    ]
    all_corners = np.concatenate(all_corners, axis=0)
    minimum = np.floor(all_corners.min(axis=0)).astype(int)
    maximum = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_width, canvas_height = (maximum - minimum).tolist()
    if (
        canvas_width <= 0
        or canvas_height <= 0
        or canvas_width * canvas_height > 100_000_000
    ):
        raise RuntimeError(
            f"Unsafe panorama canvas size: {canvas_width}x{canvas_height}."
        )
    translation = np.array(
        [[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64
    )
    warped_images = []
    warped_masks = []
    for image, mask, transform in zip(
        corrected, masks, image_to_reference.values(), strict=True
    ):
        total = translation @ transform
        warped_images.append(
            cv2.warpPerspective(
                image,
                total,
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        warped_masks.append(
            cv2.warpPerspective(
                mask,
                total,
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )

    weighted_sum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    naive = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for index in order:
        mask = (warped_masks[index] > 0).astype(np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        weights = distance * mask
        weighted_sum += warped_images[index].astype(np.float32) * weights[:, :, None]
        weight_sum += weights
        naive[mask > 0] = warped_images[index][mask > 0]
    blended = (
        np.divide(
            weighted_sum,
            np.maximum(weight_sum[:, :, None], 1e-6),
            out=np.zeros_like(weighted_sum),
            where=weight_sum[:, :, None] > 0,
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    union = (weight_sum > 0).astype(np.uint8)
    points = cv2.findNonZero(union)
    x, y, width, height = cv2.boundingRect(points)
    crop = np.s_[y : y + height, x : x + width]
    return {
        "blended": blended[crop],
        "naive": naive[crop],
        "warped_images": [image[crop] for image in warped_images],
        "warped_masks": [mask[crop] for mask in warped_masks],
        "order": order,
        "reference": reference,
        "positions": positions,
        "canvas_size": (canvas_width, canvas_height),
        "crop_bounds": (x, y, width, height),
        "tree": tree,
    }


def save_jpeg(path, image, max_side=1800, quality=94):
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_correction_figure(images, corrected, records):
    figure, axes = plt.subplots(6, 2, figsize=(12, 14), dpi=110)
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Corrected")
    for index, (before, after, record) in enumerate(
        zip(images, corrected, records, strict=True)
    ):
        axes[index, 0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        axes[index, 1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        axes[index, 0].set_ylabel(
            f"Image {index + 1}\nborder black={record['black_fraction']:.3f}"
        )
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(OUT / "q3_correction_before_after.png", bbox_inches="tight")
    plt.close(figure)


def save_heatmap(inliers):
    figure, axis = plt.subplots(figsize=(7, 6), dpi=150)
    image = axis.imshow(inliers, cmap="magma")
    axis.set_xticks(range(6), labels=[str(index) for index in range(1, 7)])
    axis.set_yticks(range(6), labels=[str(index) for index in range(1, 7)])
    axis.set_xlabel("Image")
    axis.set_ylabel("Image")
    axis.set_title("Pairwise SIFT/RANSAC inlier counts")
    for row in range(6):
        for col in range(6):
            axis.text(
                col,
                row,
                str(inliers[row, col]),
                ha="center",
                va="center",
                color="white" if inliers[row, col] < inliers.max() * 0.65 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, shrink=0.8, label="Inliers")
    figure.tight_layout()
    figure.savefig(OUT / "q3_overlap_heatmap.png", bbox_inches="tight")
    plt.close(figure)


def save_seam_comparison(result):
    i, j, _ = max(result["tree"], key=lambda edge: edge[2])
    overlap = (
        (result["warped_masks"][i] > 0) & (result["warped_masks"][j] > 0)
    ).astype(np.uint8)
    points = cv2.findNonZero(overlap)
    if points is None:
        return
    x, y, width, height = cv2.boundingRect(points)
    center_x, center_y = x + width // 2, y + height // 2
    crop_width, crop_height = min(width, 700), min(height, 500)
    x0 = max(0, min(center_x - crop_width // 2, overlap.shape[1] - crop_width))
    y0 = max(0, min(center_y - crop_height // 2, overlap.shape[0] - crop_height))
    region = np.s_[y0 : y0 + crop_height, x0 : x0 + crop_width]
    naive_crop = result["naive"][region].copy()
    blended_crop = result["blended"][region].copy()
    cv2.putText(
        naive_crop,
        "Naive overlay",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blended_crop,
        "Feather blend",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(
        str(OUT / "q3_naive_vs_blended_seam.png"), np.hstack((naive_crop, blended_crop))
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_images()
    corrected, masks, records = correct_distortions(images)
    save_correction_figure(images, corrected, records)

    original_features = extract_features(images)
    corrected_features = extract_features(corrected)
    homographies, inliers, ratios = pairwise_homographies(corrected_features)
    homography_tree = maximum_spanning_tree(homographies, inliers, ratios)
    homography_result = stitch(
        corrected, masks, homographies, inliers, ratios, homography_tree
    )
    save_jpeg(OUT / "q3_panorama_homography_preview.jpg", homography_result["blended"])

    cylindrical_images, cylindrical_masks = zip(
        *(
            cylindrical_warp(image, mask)
            for image, mask in zip(corrected, masks, strict=True)
        ),
        strict=True,
    )
    cylindrical_features = extract_features(cylindrical_images)
    translations, cylindrical_inliers, cylindrical_ratios, cylindrical_errors = (
        pairwise_cylindrical_translations(cylindrical_features)
    )
    cylindrical_tree = maximum_spanning_tree(
        translations, cylindrical_inliers, cylindrical_ratios
    )
    result = stitch(
        list(cylindrical_images),
        list(cylindrical_masks),
        translations,
        cylindrical_inliers,
        cylindrical_ratios,
        cylindrical_tree,
    )
    save_heatmap(cylindrical_inliers)
    save_seam_comparison(result)
    cv2.imwrite(str(OUT / "q3_panorama_full.png"), result["blended"])
    save_jpeg(OUT / "q3_panorama_preview.jpg", result["blended"])

    print("Detected distorted images (automatic near-black border test):")
    for record in records:
        status = "corrected" if record["distorted"] else "unchanged"
        print(
            f"  Image {record['index'] + 1}: border-connected black fraction="
            f"{record['black_fraction']:.4f}, {status}"
        )
        if record["quad"] is not None:
            print(f"    valid-content quad: {np.round(record['quad'], 1).tolist()}")
    print("Pairwise inlier counts (rows/columns are images 1–6):")
    print(inliers)
    print("Cylindrical-projection pairwise inlier counts:")
    print(cylindrical_inliers)
    print("Cylindrical pairwise inlier reprojection errors:")
    for (i, j), pair_errors in sorted(cylindrical_errors.items()):
        if cylindrical_inliers[i, j]:
            print(
                f"  {i + 1}-{j + 1}: {cylindrical_inliers[i, j]} inliers, "
                f"mean={pair_errors.mean():.3f}px, "
                f"median={np.median(pair_errors):.3f}px"
            )
    print("Cylindrical maximum spanning tree:")
    for i, j, weight in cylindrical_tree:
        print(f"  {i + 1}-{j + 1}: {weight} inliers")
    print(
        "Left-to-right order: "
        + " -> ".join(str(index + 1) for index in result["order"])
    )
    print(f"Reference image: {result['reference'] + 1}")
    print(
        f"Canvas size before empty-border crop: "
        f"{result['canvas_size'][0]}x{result['canvas_size'][1]}"
    )
    print(
        f"Panorama size after empty-border crop: "
        f"{result['blended'].shape[1]}x{result['blended'].shape[0]}"
    )
    print(
        f"Homography panorama size: "
        f"{homography_result['blended'].shape[1]}x{homography_result['blended'].shape[0]}"
    )
    print(
        "Correction sanity (dimensionless homography projectivity term on tree edges):"
    )
    sanity_edges = [
        edge
        for edge in homography_tree
        if records[edge[0]]["distorted"] or records[edge[1]]["distorted"]
    ]
    for i, j, _ in sanity_edges:
        key = (min(i, j), max(i, j))
        corrected_value = projectivity_measure(homographies[key], corrected[i].shape)
        raw_h, raw_inliers, _ = estimate_pair(
            original_features[i], original_features[j]
        )
        if raw_h is None:
            print(
                f"  {i + 1}-{j + 1}: corrected={corrected_value:.5f}; "
                "raw comparison unavailable"
            )
        else:
            raw_value = projectivity_measure(raw_h, images[i].shape)
            print(
                f"  {i + 1}-{j + 1}: raw={raw_value:.5f} ({raw_inliers} inliers), "
                f"corrected={corrected_value:.5f} ({inliers[key]} inliers)"
            )
    print(f"Output directory: {OUT}")


main()

print(f"Q3 solve runtime: {time.perf_counter() - q3_started:.2f} s")
