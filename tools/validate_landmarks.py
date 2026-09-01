"""
实测 MediaPipe FaceLandmarker 能否在手绘 mask.png 上稳定输出 468 点。

用法:
  python tools/validate_landmarks.py
  python tools/validate_landmarks.py --runs 50 --save

会尝试多种输入（原图 BGR、Alpha 合成白/灰/黑底），每种重复检测多次，
统计：是否检出、landmark 数量、多次运行坐标抖动、相邻帧（VIDEO 模式）抖动。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "face_landmarker.task"
MASK_PATH = ROOT / "assets" / "masks" / "mask.png"
OUT_DIR = ROOT / "outputs" / "landmark_test_output"

EXPECTED_LANDMARKS = 468
WINDOW_NAME = "Mask468Test"


@dataclass
class RunStats:
    name: str
    detected: int = 0
    failed: int = 0
    landmark_counts: list[int] = field(default_factory=list)
    max_jitter_px: float = 0.0
    mean_jitter_px: float = 0.0
    sample_points: list[tuple[float, float]] | None = None

    @property
    def success_rate(self) -> float:
        total = self.detected + self.failed
        return self.detected / total if total else 0.0

    @property
    def stable_count(self) -> bool:
        if not self.landmark_counts:
            return False
        return all(c == self.landmark_counts[0] for c in self.landmark_counts)


def composite_on_background(mask_bgra: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """将带 Alpha 的 mask 合成到纯色背景，得到 BGR 三通道图。"""
    h, w = mask_bgra.shape[:2]
    bg = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    if mask_bgra.shape[2] < 4:
        return mask_bgra[:, :, :3].copy()
    alpha = mask_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = mask_bgra[:, :, :3].astype(np.float32)
    return (bg.astype(np.float32) * (1.0 - alpha) + fg * alpha).astype(np.uint8)


def landmarks_to_pixels(landmarks, w: int, h: int) -> np.ndarray:
    """归一化 landmark → (N, 2) 像素坐标。"""
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)


def detect_image(landmarker: vision.FaceLandmarker, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect(mp_image)


def detect_video(landmarker: vision.FaceLandmarker, bgr: np.ndarray, timestamp_ms: int):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def jitter_between(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """两次检测同数量 landmark 的最大/平均欧氏距离（像素）。"""
    diff = np.linalg.norm(a - b, axis=1)
    return float(diff.max()), float(diff.mean())


def run_repeated_image_tests(
    landmarker: vision.FaceLandmarker,
    name: str,
    bgr: np.ndarray,
    runs: int,
) -> RunStats:
    stats = RunStats(name=name)
    h, w = bgr.shape[:2]
    ref_pixels: np.ndarray | None = None
    jitters: list[float] = []

    for _ in range(runs):
        results = detect_image(landmarker, bgr)
        if not results.face_landmarks:
            stats.failed += 1
            continue

        stats.detected += 1
        lms = results.face_landmarks[0]
        stats.landmark_counts.append(len(lms))
        pixels = landmarks_to_pixels(lms, w, h)

        if ref_pixels is None:
            ref_pixels = pixels
            stats.sample_points = [(float(x), float(y)) for x, y in pixels[:6]]
        elif pixels.shape == ref_pixels.shape:
            mx, mn = jitter_between(ref_pixels, pixels)
            jitters.append(mx)

    if jitters:
        stats.max_jitter_px = max(jitters)
        stats.mean_jitter_px = float(np.mean(jitters))
    return stats


def run_video_stability_test(
    landmarker: vision.FaceLandmarker,
    name: str,
    bgr: np.ndarray,
    frames: int,
    timestamp_ms: list[int],
) -> RunStats:
    """VIDEO 模式：同一帧 + 递增时间戳，观察跟踪是否漂移。"""
    stats = RunStats(name=f"{name} [VIDEO×{frames}]")
    h, w = bgr.shape[:2]
    ref_pixels: np.ndarray | None = None
    jitters: list[float] = []

    for _ in range(frames):
        timestamp_ms[0] += 33
        results = detect_video(landmarker, bgr, timestamp_ms[0])
        if not results.face_landmarks:
            stats.failed += 1
            continue

        stats.detected += 1
        lms = results.face_landmarks[0]
        stats.landmark_counts.append(len(lms))
        pixels = landmarks_to_pixels(lms, w, h)

        if ref_pixels is None:
            ref_pixels = pixels
        elif pixels.shape == ref_pixels.shape:
            mx, _ = jitter_between(ref_pixels, pixels)
            jitters.append(mx)

    if jitters:
        stats.max_jitter_px = max(jitters)
        stats.mean_jitter_px = float(np.mean(jitters))
    return stats


def draw_landmarks(bgr: np.ndarray, landmarks, draw_indices: bool = False) -> np.ndarray:
    """在图上绘制全部 mesh 点；可选标注前几个索引。"""
    out = bgr.copy()
    h, w = out.shape[:2]
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(out, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
        if draw_indices and i in (0, 1, 33, 61, 199, 263, 291, 467):
            cv2.putText(
                out,
                str(i),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def print_stats(stats: RunStats) -> None:
    count_info = ""
    if stats.landmark_counts:
        unique = sorted(set(stats.landmark_counts))
        count_info = f"  landmark 数量: {unique}"
        if unique == [EXPECTED_LANDMARKS]:
            count_info += " ✓"
        elif unique == [478]:
            count_info += " (含虹膜扩展点 478)"
        else:
            count_info += " ✗ 非预期"

    stable = "稳定" if stats.max_jitter_px < 0.5 and stats.detected > 1 else "有抖动"
    if stats.detected <= 1:
        stable = "样本不足"

    print(f"\n[{stats.name}]")
    print(f"  检出率: {stats.success_rate * 100:.1f}% ({stats.detected}/{stats.detected + stats.failed})")
    print(count_info)
    if stats.detected > 1:
        print(f"  坐标抖动: max={stats.max_jitter_px:.3f}px  mean={stats.mean_jitter_px:.3f}px  → {stable}")


def build_landmarker(mode: vision.RunningMode) -> vision.FaceLandmarker:
    if not MODEL_PATH.exists():
        print(f"错误：找不到 {MODEL_PATH.name}")
        print(
            "下载: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        raise SystemExit(1)

    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mode,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 MediaPipe 对 mask.png 的 468 点检测稳定性")
    parser.add_argument("--runs", type=int, default=30, help="每种输入 IMAGE 模式重复次数")
    parser.add_argument("--video-frames", type=int, default=60, help="VIDEO 模式重复帧数")
    parser.add_argument("--save", action="store_true", help="保存可视化图与 JSON 报告")
    parser.add_argument("--show", action="store_true", help="弹窗显示最佳检出结果")
    args = parser.parse_args()

    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"错误：找不到 {MASK_PATH}")
        raise SystemExit(1)

    h, w = mask.shape[:2]
    print(f"Mask: {MASK_PATH.name}  尺寸 {w}×{h}  通道 {mask.shape[2]}")
    print(f"期望 landmark 数: {EXPECTED_LANDMARKS}")
    print(f"IMAGE 重复: {args.runs} 次  |  VIDEO 帧: {args.video_frames} 次")

    variants: list[tuple[str, np.ndarray]] = [
        ("BGR 丢弃 Alpha", mask[:, :, :3].copy() if mask.shape[2] >= 3 else mask.copy()),
        ("白底合成", composite_on_background(mask, (255, 255, 255))),
        ("灰底合成", composite_on_background(mask, (128, 128, 128))),
        ("黑底合成", composite_on_background(mask, (0, 0, 0))),
    ]

    image_landmarker = build_landmarker(vision.RunningMode.IMAGE)
    video_landmarker = build_landmarker(vision.RunningMode.VIDEO)
    video_timestamp_ms = [0]  # VIDEO 模式要求跨调用单调递增

    all_stats: list[RunStats] = []
    best_vis: np.ndarray | None = None
    best_count = -1

    try:
        for name, bgr in variants:
            stats = run_repeated_image_tests(image_landmarker, name, bgr, args.runs)
            all_stats.append(stats)
            print_stats(stats)

            if stats.detected > 0:
                results = detect_image(image_landmarker, bgr)
                if results.face_landmarks:
                    n = len(results.face_landmarks[0])
                    if n > best_count:
                        best_count = n
                        best_vis = draw_landmarks(bgr, results.face_landmarks[0], draw_indices=True)

            if stats.detected > 0:
                vstats = run_video_stability_test(
                    video_landmarker, name, bgr, args.video_frames, video_timestamp_ms
                )
                all_stats.append(vstats)
                print_stats(vstats)

        # 汇总
        print("\n" + "=" * 60)
        print("汇总")
        any_detect = any(s.detected > 0 for s in all_stats if "[VIDEO" not in s.name)
        if not any_detect:
            print("结论: MediaPipe 未能在这张手绘 mask 上检出人脸，468 点不可用。")
        else:
            image_stats = [s for s in all_stats if "[VIDEO" not in s.name and s.detected > 0]
            counts = sorted({c for s in image_stats for c in s.landmark_counts})
            ok_468 = counts == [EXPECTED_LANDMARKS]
            stable_runs = all(
                s.max_jitter_px < 0.5 for s in all_stats if s.detected > 1
            )
            print(f"  可检出变体: {sum(1 for s in image_stats if s.detected > 0)}/{len(image_stats)}")
            print(f"  landmark 数量: {counts}")
            if ok_468:
                print("  数量符合 468 ✓")
            elif counts == [478]:
                print("  模型返回 478 点（468 mesh + 虹膜），可截取前 468 点使用")
            print(
                f"  多次运行坐标稳定: {'是' if stable_runs else '否'}"
                f"（阈值 0.5px）"
            )

        if args.save and OUT_DIR:
            OUT_DIR.mkdir(exist_ok=True)
            report = {
                "mask": str(MASK_PATH),
                "mask_size": [w, h],
                "expected_landmarks": EXPECTED_LANDMARKS,
                "runs_per_variant": args.runs,
                "results": [
                    {
                        "name": s.name,
                        "success_rate": s.success_rate,
                        "detected": s.detected,
                        "failed": s.failed,
                        "landmark_counts": s.landmark_counts,
                        "max_jitter_px": s.max_jitter_px,
                        "mean_jitter_px": s.mean_jitter_px,
                    }
                    for s in all_stats
                ],
            }
            report_path = OUT_DIR / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存: {report_path}")

            if best_vis is not None:
                vis_path = OUT_DIR / "landmarks_overlay.png"
                cv2.imwrite(str(vis_path), best_vis)
                print(f"可视化已保存: {vis_path}")

        if args.show and best_vis is not None:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME, best_vis)
            print("\n按任意键关闭窗口…")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    finally:
        image_landmarker.close()
        video_landmarker.close()


if __name__ == "__main__":
    main()
