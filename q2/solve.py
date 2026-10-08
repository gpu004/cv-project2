import time
from pathlib import Path
from typing import Any as Q2Any
from typing import cast

import cv2
import numpy as np

q2_started = time.perf_counter()

DATA = Path("data/q2")
OUT = Path("outputs/q2")


def cross2d(first, second):
    return first[0] * second[1] - first[1] * second[0]


def resize_long_side(image, limit):
    scale = min(1.0, limit / max(image.shape[:2]))
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def estimate_homography(source, scene):
    source_small, source_scale = resize_long_side(source, 1200)
    scene_small, scene_scale = resize_long_side(scene, 2000)
    source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_small, cv2.COLOR_BGR2GRAY)
    sift = cast(Q2Any, cv2).SIFT_create(nfeatures=7000)
    keypoints_source, descriptors_source = sift.detectAndCompute(source_gray, None)
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_gray, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        descriptors_source, descriptors_scene, k=2
    )
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        raise RuntimeError(
            f"Only {len(good)} ratio-test matches; at least four are required."
        )

    points_source = np.asarray(
        [keypoints_source[m.queryIdx].pt for m in good], dtype=np.float32
    )
    points_scene = np.asarray(
        [keypoints_scene[m.trainIdx].pt for m in good], dtype=np.float32
    )
    homography, inlier_mask = cv2.findHomography(
        points_source,
        points_scene,
        cv2.USAC_MAGSAC,
        3.0,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed.")

    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(points_source[inliers, None, :], homography)[
        :, 0
    ]
    errors = np.linalg.norm(projected - points_scene[inliers], axis=1)
    source_to_small = np.diag([source_scale, source_scale, 1.0])
    scene_to_full = np.diag([1.0 / scene_scale, 1.0 / scene_scale, 1.0])
    homography_full = scene_to_full @ homography @ source_to_small
    return {
        "homography": homography_full,
        "source_small": source_small,
        "scene_small": scene_small,
        "keypoints_source": keypoints_source,
        "keypoints_scene": keypoints_scene,
        "good": good,
        "inliers": inliers,
        "errors": errors,
        "scene_scale": scene_scale,
    }


def line_intersection(first, second):
    point_a, direction_a = first
    point_b, direction_b = second
    denominator = cross2d(direction_a, direction_b)
    if abs(denominator) < 1e-6:
        raise RuntimeError("Refined cover edges are nearly parallel.")
    scale = cross2d(point_b - point_a, direction_b) / denominator
    return point_a + scale * direction_a


def refine_quad_edges(scene, corners):
    lab = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), sigmaX=2.0, sigmaY=2.0)
    offsets = np.arange(-20, 103, dtype=np.float32)
    side_names = ("top", "right", "bottom", "left")
    center = corners.mean(axis=0)
    lines = []
    diagnostics = []

    for index, side_name in enumerate(side_names):
        start = corners[index]
        end = corners[(index + 1) % 4]
        direction = end - start
        length = np.linalg.norm(direction)
        tangent = direction / max(length, 1e-6)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        if np.dot(normal, center - (start + end) / 2) > 0:
            normal = -normal
        fractions = np.linspace(0.08, 0.92, 60, dtype=np.float32)
        samples = start[None, :] + fractions[:, None] * direction
        map_x = (samples[:, None, 0] + offsets[None, :] * normal[0]).astype(np.float32)
        map_y = (samples[:, None, 1] + offsets[None, :] * normal[1]).astype(np.float32)
        profiles = cv2.remap(
            lab, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
        )
        mean_profile = profiles.mean(axis=0)
        std_profile = profiles.std(axis=0)

        candidate_offsets = np.arange(0, 91, dtype=np.int32)
        jump_profile = np.zeros(len(candidate_offsets), dtype=np.float32)
        for candidate_index, offset in enumerate(candidate_offsets):
            before = mean_profile[
                offset - 12 - int(offsets[0]) : offset - 3 - int(offsets[0]) + 1
            ]
            after = mean_profile[
                offset + 3 - int(offsets[0]) : offset + 12 - int(offsets[0]) + 1
            ]
            jump_profile[candidate_index] = np.linalg.norm(
                after.mean(axis=0) - before.mean(axis=0)
            )

        best_index = int(np.argmax(jump_profile))
        best_offset = int(candidate_offsets[best_index])
        baseline_jump = float(jump_profile[0])
        best_jump = float(jump_profile[best_index])
        if best_offset > 0 and best_jump < 1.2 * baseline_jump:
            best_offset = 0
            best_jump = baseline_jump

        gradient = np.linalg.norm(profiles[:, 1:] - profiles[:, :-1], axis=2)
        gradient_offsets = (offsets[1:] + offsets[:-1]) / 2
        local = (gradient_offsets >= best_offset - 6) & (
            gradient_offsets <= best_offset + 6
        )
        local_indices = np.flatnonzero(local)
        boundary_points = []
        local_offsets = []
        for sample_index, sample in enumerate(samples):
            profile_gradient = gradient[sample_index, local]
            best_gradient_index = local_indices[int(np.argmax(profile_gradient))]
            boundary_offset = float(gradient_offsets[best_gradient_index])
            boundary_points.append(sample + boundary_offset * normal)
            local_offsets.append(boundary_offset)

        fitted = cv2.fitLine(
            np.asarray(boundary_points, dtype=np.float32).reshape(-1, 1, 2),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).ravel()
        line = (
            np.array([fitted[2], fitted[3]], dtype=np.float32),
            np.array([fitted[0], fitted[1]], dtype=np.float32),
        )
        lines.append(line)
        diagnostics.append(
            {
                "name": side_name,
                "offset": best_offset,
                "jump": best_jump,
                "baseline_jump": baseline_jump,
                "std": float(np.linalg.norm(std_profile.mean(axis=0))),
                "local_offset_median": float(np.median(local_offsets)),
            }
        )

    refined = np.asarray(
        [line_intersection(lines[(index - 1) % 4], lines[index]) for index in range(4)],
        dtype=np.float32,
    )
    if not cv2.isContourConvex(np.round(refined).astype(np.int32)):
        raise RuntimeError("Refined cover boundary is not a convex quadrilateral.")
    return refined, diagnostics


def save_preview(path, image, max_side=1800, quality=93):
    preview, _ = resize_long_side(image, max_side)
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), preview, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), preview)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = cv2.imread(str(DATA / "Scene.png"))
    source = cv2.imread(str(DATA / "Source_Screen.png"))
    replacement = cv2.imread(str(DATA / "replacement_screen.jpg"))
    if scene is None or source is None or replacement is None:
        raise FileNotFoundError(f"Could not load all Q2 inputs from {DATA}.")

    result = estimate_homography(source, scene)
    source_h, source_w = source.shape[:2]
    replacement_h, replacement_w = replacement.shape[:2]
    source_corners = np.asarray(
        [[0, 0], [source_w - 1, 0], [source_w - 1, source_h - 1], [0, source_h - 1]],
        dtype=np.float32,
    )
    detected_corners = cv2.perspectiveTransform(
        source_corners[None, :, :], result["homography"]
    )[0]
    refined_corners, side_diagnostics = refine_quad_edges(scene, detected_corners)
    replacement_corners = np.asarray(
        [
            [0, 0],
            [replacement_w - 1, 0],
            [replacement_w - 1, replacement_h - 1],
            [0, replacement_h - 1],
        ],
        dtype=np.float32,
    )
    replacement_to_scene = cv2.getPerspectiveTransform(
        replacement_corners, refined_corners
    )
    replacement_warped = cv2.warpPerspective(
        replacement,
        replacement_to_scene,
        (scene.shape[1], scene.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask = np.zeros(scene.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(
        mask, np.round(refined_corners).astype(np.int32), 255, lineType=cv2.LINE_AA
    )
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.5)
    alpha = mask.astype(np.float32)[:, :, None] / 255.0
    augmented = np.clip(
        replacement_warped.astype(np.float32) * alpha
        + scene.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)

    good = result["good"]
    inlier_matches = [
        match for match, keep in zip(good, result["inliers"], strict=True) if keep
    ]
    match_vis = cast(Q2Any, cv2).drawMatches(
        result["source_small"],
        result["keypoints_source"],
        result["scene_small"],
        result["keypoints_scene"],
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_preview(OUT / "q2_inlier_matches.jpg", match_vis, max_side=2000)

    scene_outline = scene.copy()
    cv2.polylines(
        scene_outline,
        [np.round(detected_corners).astype(np.int32)],
        True,
        (0, 255, 0),
        8,
        cv2.LINE_AA,
    )
    save_preview(OUT / "q2_detected_quad.jpg", scene_outline, max_side=1800)

    all_corners = np.vstack((detected_corners, refined_corners))
    margin = 80
    crop_x0 = max(0, int(np.floor(all_corners[:, 0].min())) - margin)
    crop_x1 = min(scene.shape[1], int(np.ceil(all_corners[:, 0].max())) + margin)
    crop_y0 = max(0, int(np.floor(all_corners[:, 1].min())) - margin)
    crop_y1 = min(scene.shape[0], int(np.ceil(all_corners[:, 1].max())) + margin)
    refinement_crop = scene[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    local_detected = np.round(detected_corners - [crop_x0, crop_y0]).astype(np.int32)
    local_refined = np.round(refined_corners - [crop_x0, crop_y0]).astype(np.int32)
    cv2.polylines(
        refinement_crop, [local_detected], True, (255, 255, 0), 6, cv2.LINE_AA
    )
    cv2.polylines(refinement_crop, [local_refined], True, (0, 0, 255), 6, cv2.LINE_AA)
    cv2.putText(
        refinement_crop,
        "SIFT projection",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        refinement_crop,
        "Refined boundary",
        (20, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    save_preview(OUT / "q2_refinement.jpg", refinement_crop, max_side=1800)

    cv2.imwrite(str(OUT / "q2_augmented_full.png"), augmented)
    save_preview(OUT / "q2_augmented_preview.jpg", augmented, max_side=1800)

    corner_names = ("tl", "tr", "br", "bl")
    crop_radius = 300
    for name, old_point, new_point in zip(
        corner_names, detected_corners, refined_corners, strict=True
    ):
        point = (old_point + new_point) / 2
        x, y = np.round(point).astype(int)
        x0, x1 = max(0, x - crop_radius), min(scene.shape[1], x + crop_radius)
        y0, y1 = max(0, y - crop_radius), min(scene.shape[0], y + crop_radius)
        original_crop = scene[y0:y1, x0:x1].copy()
        result_crop = augmented[y0:y1, x0:x1].copy()
        cv2.putText(
            original_crop,
            "Original",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            original_crop,
            "Original",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            result_crop,
            "Replacement",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result_crop,
            "Replacement",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        comparison = cv2.hconcat((original_crop, result_crop))
        cv2.imwrite(
            str(OUT / f"q2_corner_{name}.jpg"),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    errors = result["errors"]
    print(f"Source keypoints: {len(result['keypoints_source'])}")
    print(f"Scene keypoints: {len(result['keypoints_scene'])}")
    print(f"Good matches: {len(good)}")
    print(f"Inliers: {int(result['inliers'].sum())}")
    print(f"Inlier ratio: {result['inliers'].mean():.4f}")
    print(
        f"Inlier reprojection error (scaled match pixels): mean={errors.mean():.3f}, "
        f"median={np.median(errors):.3f}"
    )
    print("Projected source-cover corners (full-resolution scene x,y):")
    for name, point in zip(corner_names, detected_corners, strict=True):
        print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")
    print("Per-side outward color-profile boundary refinement:")
    for diagnostic in side_diagnostics:
        print(
            f"  {diagnostic['name']}: t*={diagnostic['offset']} px, "
            f"Lab jump={diagnostic['jump']:.2f}, "
            f"t=0 jump={diagnostic['baseline_jump']:.2f}, "
            f"along-side Lab std={diagnostic['std']:.2f}, "
            f"median local edge={diagnostic['local_offset_median']:.1f} px"
        )
    print("Old vs refined corners (full-resolution scene x,y):")
    for name, old_point, new_point in zip(
        corner_names, detected_corners, refined_corners, strict=True
    ):
        shift = np.linalg.norm(new_point - old_point)
        print(
            f"  {name}: ({old_point[0]:.1f}, {old_point[1]:.1f}) -> "
            f"({new_point[0]:.1f}, {new_point[1]:.1f}); "
            f"corner displacement={shift:.1f} px"
        )
    print(
        "Per-side outward offsets (top→right→bottom→left, px): "
        f"{[d['offset'] for d in side_diagnostics]}"
    )
    print(f"Output directory: {OUT}")


main()

print(f"Q2 solve runtime: {time.perf_counter() - q2_started:.2f} s")
