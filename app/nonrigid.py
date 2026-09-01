"""
阿格尼痛苦脸实时映射特效（v2：非刚性变形版）

相比 v1（main.py）用 6 个点做一次全局相似变换，v2 改用 MediaPipe 输出的
468 个人脸关键点，对人脸做 Delaunay 三角剖分，再逐三角形做仿射变换
（分片仿射 / piecewise affine），用"分段线性"逼近脸部的局部非刚性形变，
从而解决 v1"全局仿射无法跟随眉毛、嘴角等局部运动"的局限。

整体流程：
  启动: --mask 指定图片 → MediaPipe(IMAGE) 检测 468 点 → 按 mask 写入 cache/
        → Delaunay 三角剖分（拓扑与关键点一并缓存，每个 mask 独立一份）
  每帧: 摄像头帧 → MediaPipe(VIDEO) 检测 468 点 → 按编号一一对应
        → 逐三角形 getAffineTransform + warpAffine + 三角形掩膜（包围盒 ROI 加速）
        → alpha 羽化 → Alpha 混合叠加 → 显示

模型从项目根目录加载，mask 图从 assets/masks 加载。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== 路径与常量 ==================

# 本脚本所在目录（app/）
BASE_DIR = Path(__file__).resolve().parent
# 项目根目录（父目录）
ROOT = BASE_DIR.parent

MODEL_PATH = ROOT / "face_landmarker.task"
MASK_DIR = ROOT / "assets" / "masks"
DEFAULT_MASK_PATH = MASK_DIR / "mask.png"
# 每个 mask 独立缓存：468 点 + Delaunay 三角拓扑
CACHE_DIR = ROOT / ".cache" / "nonrigid"

# 只取前 468 个 mesh 点；模型实际会返回 478 点（468 mesh + 10 虹膜），
# 虹膜点位于眼内、彼此过近，易在剖分中产生退化三角形，故剔除。
NUM_POINTS = 468

WINDOW_NAME = "AgniPainFaceV2"


# ================== mask 路径与缓存 ==================


def resolve_mask_path(mask_arg: str) -> Path:
    """解析 --mask：文件名从 assets/masks 查找，其余相对项目根目录。"""
    path = Path(mask_arg)
    if not path.is_absolute():
        path = MASK_DIR / path if len(path.parts) == 1 else ROOT / path
    return path.resolve()


def cache_path_for_mask(mask_path: Path) -> Path:
    """.cache/nonrigid/mask_landmarks_{stem}.json"""
    return CACHE_DIR / f"mask_landmarks_{mask_path.stem}.json"


def resolve_mask_bg_mode(mask_path: Path, bg_arg: str) -> str:
    """
    决定检测 mask 关键点时的背景合成方式。
    auto: 线稿痛苦脸 mask.png 用白底，其余（如照片类 mask1）用 bgr 直读。
    """
    if bg_arg != "auto":
        return bg_arg
    return "white" if mask_path.stem == "mask" else "bgr"


def set_opencv_window_title(window_name: str, display_title: str) -> bool:
    """用 Win32 SetWindowTextW 设置窗口标题，正确显示中文（仅 Windows）。"""
    if sys.platform != "win32":
        return False

    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:
        user32.SetWindowTextW(hwnd, display_title)
        return True
    return False


# ================== MediaPipe 检测 ==================


def build_landmarker(mode: vision.RunningMode, min_conf: float) -> vision.FaceLandmarker:
    """创建指定运行模式的 FaceLandmarker。"""
    if not MODEL_PATH.exists():
        print(f"错误：找不到 {MODEL_PATH}")
        print(
            "下载地址：https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        raise SystemExit(1)

    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mode,
            num_faces=1,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
    )


def landmarks_to_pixels(landmarks, w: int, h: int) -> np.ndarray:
    """归一化 landmark（前 NUM_POINTS 个）→ (NUM_POINTS, 2) 像素坐标。"""
    pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in range(NUM_POINTS)]
    return np.array(pts, dtype=np.float32)


def composite_on_background(mask_bgra: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """将带 Alpha 的 mask 合成到纯色背景，得到 BGR 三通道图。"""
    if mask_bgra.shape[2] < 4:
        return mask_bgra[:, :, :3].copy()
    h, w = mask_bgra.shape[:2]
    bg = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    alpha = mask_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = mask_bgra[:, :, :3].astype(np.float32)
    return (bg.astype(np.float32) * (1.0 - alpha) + fg * alpha).astype(np.uint8)


def mask_bgr_for_detection(mask_bgra: np.ndarray, bg_mode: str) -> np.ndarray:
    """按 bg_mode 生成送入 MediaPipe IMAGE 检测的 BGR 图。"""
    if bg_mode == "bgr":
        return mask_bgra[:, :, :3].copy() if mask_bgra.shape[2] >= 3 else mask_bgra.copy()
    if bg_mode == "white":
        return composite_on_background(mask_bgra, (255, 255, 255))
    if bg_mode == "black":
        return composite_on_background(mask_bgra, (0, 0, 0))
    raise ValueError(f"未知 mask-bg 模式: {bg_mode}")


def detect_mask_landmarks(mask_bgra: np.ndarray, mask_path: Path, bg_mode: str) -> np.ndarray:
    """对 mask 用 IMAGE 模式检测一次 468 点（mask 像素坐标）。"""
    h, w = mask_bgra.shape[:2]
    bgr = mask_bgr_for_detection(mask_bgra, bg_mode)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    landmarker = build_landmarker(vision.RunningMode.IMAGE, min_conf=0.3)
    try:
        results = landmarker.detect(mp_image)
    finally:
        landmarker.close()

    if not results.face_landmarks:
        raise RuntimeError(
            f"MediaPipe 未能在 {mask_path.name} 上检测到人脸（bg={bg_mode}）。"
            f"可尝试 --mask-bg white/black/bgr，或用 tools/validate_landmarks.py 预检。"
        )
    return landmarks_to_pixels(results.face_landmarks[0], w, h)


def load_or_build_mask_assets(
    mask_bgra: np.ndarray,
    mask_path: Path,
    bg_mode: str,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    加载或生成 mask 的 468 点与 Delaunay 三角拓扑，写入 per-mask 缓存。

    缓存路径: .cache/nonrigid/mask_landmarks_{stem}.json
    """
    h, w = mask_bgra.shape[:2]
    cache_path = cache_path_for_mask(mask_path)
    mask_key = str(mask_path.relative_to(ROOT)) if mask_path.is_relative_to(ROOT) else str(mask_path)

    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            pts = np.array(data["points"], dtype=np.float32)
            tris_raw = data.get("triangles")
            if (
                pts.shape == (NUM_POINTS, 2)
                and data.get("mask_size") == [w, h]
                and tris_raw is not None
                and len(tris_raw) > 0
            ):
                triangles = [tuple(t) for t in tris_raw]
                print(f"已加载缓存：{cache_path.relative_to(ROOT)}（{len(triangles)} 个三角形）")
                return pts, triangles
            print("缓存与当前 mask 不匹配或缺少拓扑，重新检测与剖分。")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            print("缓存文件损坏，重新检测与剖分。")

    print(f"检测 {mask_path.name} 的 468 关键点（bg={bg_mode}）……")
    pts = detect_mask_landmarks(mask_bgra, mask_path, bg_mode)
    triangles = build_triangulation(pts, (w, h))
    print(f"Delaunay 三角剖分：{len(triangles)} 个三角形")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mask_path": mask_key,
                "mask_size": [w, h],
                "num_points": NUM_POINTS,
                "bg_mode": bg_mode,
                "points": pts.tolist(),
                "triangles": [list(t) for t in triangles],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"已写入缓存：{cache_path.relative_to(ROOT)}")
    return pts, triangles


def get_dense_landmarks(landmarker, image, timestamp_ms: int) -> np.ndarray | None:
    """从一帧图像检测人脸，返回 (NUM_POINTS, 2) 像素坐标；未检测到返回 None。"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not results.face_landmarks:
        return None
    h, w = image.shape[:2]
    return landmarks_to_pixels(results.face_landmarks[0], w, h)


# ================== 三角剖分与分片仿射 ==================


def build_triangulation(points: np.ndarray, size: tuple[int, int]) -> list[tuple[int, int, int]]:
    """
    对点集做一次 Delaunay 三角剖分，返回三角形的“点编号三元组”列表（拓扑）。

    返回的是编号 (i, j, k) 而非坐标：mask 与人脸点集编号语义一致，
    因此该拓扑可在两侧共用——mask 静止，启动时算一次即可。
    """
    w, h = size
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Subdiv2D 要求插入点严格落在 rect 内，做一次裁剪兜底
    clamped = np.empty_like(points)
    for i, (x, y) in enumerate(points):
        px = float(min(max(x, 0), w - 1))
        py = float(min(max(y, 0), h - 1))
        clamped[i] = (px, py)
        subdiv.insert((px, py))

    triangles: list[tuple[int, int, int]] = []
    for t in subdiv.getTriangleList():
        verts = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        # 过滤掉触及外接“虚拟点”（落在 rect 外）的三角形
        if (
            np.any(verts[:, 0] < 0)
            or np.any(verts[:, 0] > w)
            or np.any(verts[:, 1] < 0)
            or np.any(verts[:, 1] > h)
        ):
            continue

        idx: list[int] = []
        ok = True
        for v in verts:
            dist = np.linalg.norm(clamped - v, axis=1)
            j = int(np.argmin(dist))
            if dist[j] > 1.0:  # 顶点不属于我们的点集
                ok = False
                break
            idx.append(j)

        if ok and len(set(idx)) == 3:
            triangles.append(tuple(idx))

    return triangles


def warp_piecewise(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    triangles: list[tuple[int, int, int]],
    out_size: tuple[int, int],
) -> np.ndarray:
    """
    逐三角形把 src_img 变形对齐到 dst_pts 描述的人脸，累积输出非刚性变形后的图。

    每个三角形单独求仿射（getAffineTransform），并仅在其“目标包围盒”ROI 内做
    warpAffine 与三角形掩膜填充，避免对整张图变换，以维持实时帧率。
    """
    out_w, out_h = out_size
    channels = src_img.shape[2]
    src_h, src_w = src_img.shape[:2]
    output = np.zeros((out_h, out_w, channels), dtype=src_img.dtype)

    for tri in triangles:
        idx = list(tri)
        src_tri = src_pts[idx].astype(np.float32)
        dst_tri = dst_pts[idx].astype(np.float32)

        # 目标三角形包围盒，并立即与画面相交。
        # 关键：侧脸时 MediaPipe 可能把点外推到画面外极远处，包围盒会非常大；
        # 必须先裁剪到画面范围再决定 warpAffine 的输出尺寸，否则会分配超大图像导致崩溃。
        dx, dy, dw, dh = cv2.boundingRect(dst_tri)
        if dw <= 0 or dh <= 0:
            continue
        ix0, iy0 = max(dx, 0), max(dy, 0)
        ix1, iy1 = min(dx + dw, out_w), min(dy + dh, out_h)
        rw, rh = ix1 - ix0, iy1 - iy0
        if rw <= 0 or rh <= 0:  # 三角形完全在画面外
            continue

        # 源三角形包围盒（裁剪到 mask 范围内）
        sx, sy, sw, sh = cv2.boundingRect(src_tri)
        sx0, sy0 = max(sx, 0), max(sy, 0)
        sx1, sy1 = min(sx + sw, src_w), min(sy + sh, src_h)
        if sx1 <= sx0 or sy1 <= sy0:
            continue

        src_local = (src_tri - [sx0, sy0]).astype(np.float32)
        # 目标顶点相对“裁剪后区域”的局部坐标——warp 输出尺寸即为 (rw, rh)，恒不超过画面
        dst_local = (dst_tri - [ix0, iy0]).astype(np.float32)
        src_crop = src_img[sy0:sy1, sx0:sx1]

        matrix = cv2.getAffineTransform(src_local, dst_local)
        warped = cv2.warpAffine(
            src_crop,
            matrix,
            (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        tri_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, np.int32(dst_local), 255, cv2.LINE_AA)
        region_mask = tri_mask > 0
        roi = output[iy0:iy1, ix0:ix1]
        roi[region_mask] = warped[region_mask]

    return output


# ================== 合成与显示 ==================


def feather_alpha(warped_bgra: np.ndarray, ksize: int = 5) -> np.ndarray:
    """对 alpha 通道做轻微高斯模糊，羽化分片拼接与贴图边界硬边。"""
    if warped_bgra.shape[2] < 4:
        return warped_bgra
    k = ksize if ksize % 2 == 1 else ksize + 1
    out = warped_bgra.copy()
    out[:, :, 3] = cv2.GaussianBlur(out[:, :, 3], (k, k), 0)
    return out


def overlay_mask(frame: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
    """按 Alpha 把变形后的 mask 叠加到画面：结果 = 背景×(1-α) + 前景×α。"""
    if warped_mask.shape[2] == 4:
        alpha = warped_mask[:, :, 3:4] / 255.0
        foreground = warped_mask[:, :, :3].astype(float)
        background = frame.astype(float)
        return (background * (1.0 - alpha) + foreground * alpha).astype(np.uint8)
    return cv2.addWeighted(frame, 0.7, warped_mask, 0.3, 0)


def draw_mesh(frame: np.ndarray, pts: np.ndarray, triangles: list[tuple[int, int, int]]) -> None:
    """在画面上叠加三角网格（便于截图展示剖分结构）。"""
    for tri in triangles:
        poly = pts[list(tri)].astype(np.int32)
        cv2.polylines(frame, [poly], True, (0, 200, 0), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    show_mesh: bool,
    detected: bool,
    mask_name: str,
) -> None:
    """左上角叠加 FPS、当前 mask 与操作提示。"""
    lines = [
        f"FPS: {fps:4.1f}   mask: {mask_name}",
        f"mesh: {'ON' if show_mesh else 'off'}   keys: [t] mesh  [q] quit",
        "FACE DETECTED" if detected else "no face",
    ]
    x0, y0, line_h = 10, 10, 22
    box_w, box_h = 420, line_h * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (200, 200, 200)
        if i == 2:
            color = (160, 255, 160) if detected else (160, 160, 255)
        cv2.putText(
            frame,
            line,
            (x0, y0 + 16 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


# ================== 主程序 ==================


def main() -> None:
    parser = argparse.ArgumentParser(description="人脸替换 v2：非刚性变形（Delaunay + 分片仿射）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引（默认 0）")
    parser.add_argument(
        "--mask",
        default="mask.png",
        help="mask 文件名、相对项目根目录的路径或绝对路径（默认 mask.png）",
    )
    parser.add_argument(
        "--mask-bg",
        choices=("auto", "white", "black", "bgr"),
        default="auto",
        help="检测 mask 关键点时的背景：auto=mask.png 用白底、其余用 bgr",
    )
    parser.add_argument("--mesh", action="store_true", help="启动即叠加三角网格")
    parser.add_argument("--no-feather", action="store_true", help="关闭 alpha 羽化")
    parser.add_argument("--feather-ksize", type=int, default=5, help="羽化高斯核大小（奇数）")
    args = parser.parse_args()

    mask_path = resolve_mask_path(args.mask)
    if not mask_path.exists():
        print(f"错误：找不到 mask 文件 {mask_path}")
        raise SystemExit(1)

    bg_mode = resolve_mask_bg_mode(mask_path, args.mask_bg)
    window_title = f"Face Replace v2 - {mask_path.stem}"

    # mask（保留 BGRA 以使用其 Alpha 透明通道）
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"错误：无法读取 {mask_path}")
        raise SystemExit(1)
    if mask.shape[2] == 3:  # 无 Alpha 时补一个全不透明通道
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
    mask_h, mask_w = mask.shape[:2]
    print(f"Mask: {mask_path.name}  尺寸 {mask_w}×{mask_h}  通道 {mask.shape[2]}  检测 bg={bg_mode}")

    mask_pts, triangles = load_or_build_mask_assets(mask, mask_path, bg_mode)

    face_landmarker = build_landmarker(vision.RunningMode.VIDEO, min_conf=0.5)

    # Windows 上用 DSHOW 后端，开/关摄像头更可靠（避免句柄残留导致下次打不开）
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        print("提示：若刚崩溃过，可能有残留进程仍占用摄像头；可在任务管理器结束 python 进程后重试，或换 --camera 索引。")
        face_landmarker.close()
        raise SystemExit(1)
    # 降低分辨率以提升分片仿射的实时帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    title_applied = False
    show_mesh = args.mesh
    do_feather = not args.no_feather

    start_time = time.time()
    prev_t = start_time
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            timestamp_ms = int((time.time() - start_time) * 1000)
            face_pts = get_dense_landmarks(face_landmarker, frame, timestamp_ms)

            output = frame
            detected = face_pts is not None
            if detected:
                try:
                    warped = warp_piecewise(
                        mask, mask_pts, face_pts, triangles, (frame.shape[1], frame.shape[0])
                    )
                    if do_feather:
                        warped = feather_alpha(warped, args.feather_ksize)
                    output = overlay_mask(frame, warped)
                    if show_mesh:
                        draw_mesh(output, face_pts, triangles)
                except cv2.error as exc:  # 单帧异常不应使整个程序崩溃
                    print(f"\n[warn] 本帧变形失败，已跳过：{exc}")
                    output = frame

            # FPS（指数平滑）
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            draw_hud(output, fps, show_mesh, detected, mask_path.name)
            cv2.imshow(WINDOW_NAME, output)

            if not title_applied:
                title_applied = set_opencv_window_title(WINDOW_NAME, window_title)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                show_mesh = not show_mesh

            # 检测用户点窗口 X 关闭：否则进程会在后台继续占用摄像头
            if frame_count > 1 and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        face_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
