"""
阿格尼痛苦脸实时映射特效（基础版）

整体流程：
  摄像头帧 → MediaPipe 检测人脸关键点 → 计算 mask 到人脸的仿射变换
  → 将 mask 变形对齐 → Alpha 混合叠加 → 显示结果
"""

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

# 项目根目录，用于定位模型和 mask 文件（避免因工作目录不同而找不到文件）
ROOT = Path(__file__).resolve().parent.parent
# MediaPipe 人脸关键点检测模型（需提前下载到项目目录）
MODEL_PATH = ROOT / "face_landmarker.task"
# 阿格尼痛苦脸 PNG，需带 Alpha 透明通道
MASK_PATH = ROOT / "assets" / "masks" / "mask.png"
# 由 tools/annotate_mask.py 标注后生成的关键点配置（存在则优先使用）
MASK_POINTS_PATH = ROOT / "assets" / "masks" / "mask_points.json"

# MediaPipe 人脸网格中的关键点索引，与 build_mask_points 中的 6 个点一一对应
# 33=左眼中心, 263=右眼中心, 1=鼻尖, 61=左嘴角, 291=右嘴角, 199=下巴
LANDMARK_INDICES = [33, 263, 1, 61, 291, 199]

# OpenCV 内部窗口名（仅 ASCII）；显示标题单独用 Win32 API 设置 UTF-16 中文
WINDOW_NAME = "AgniPainFace"
WINDOW_TITLE = "Agni Pain Face - 基础版"


def set_opencv_window_title(window_name: str, display_title: str) -> bool:
    """
    用 Win32 SetWindowTextW 设置窗口标题，正确显示 UTF-8 中文。
    OpenCV 自带的 namedWindow/imshow 在 Windows 上无法可靠显示中文标题。
    """
    if sys.platform != "win32":
        return False

    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:
        user32.SetWindowTextW(hwnd, display_title)
        return True
    return False


# ================== 初始化 ==================

# 检查模型文件是否存在，不存在则提示下载地址并退出
if not MODEL_PATH.exists():
    print(f"错误：找不到 {MODEL_PATH.name}，请先下载模型文件。")
    print("下载地址：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    raise SystemExit(1)

# 配置 MediaPipe FaceLandmarker
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # 视频/摄像头模式，需传入时间戳
        num_faces=1,                              # 只跟踪一张脸
        min_face_detection_confidence=0.5,        # 人脸检测置信度阈值
        min_face_presence_confidence=0.5,           # 人脸存在置信度阈值
        min_tracking_confidence=0.5,                # 跟踪置信度阈值
    )
)

# 读取 mask 图片；IMREAD_UNCHANGED 保留 Alpha 通道（BGRA 四通道）
mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_UNCHANGED)
if mask is None:
    print(f"错误：找不到 {MASK_PATH.name}！")
    raise SystemExit(1)

print(f"Mask 尺寸: {mask.shape}")  # 例如 (570, 438, 4) 表示高×宽×通道
print(
    "仿射变换公式: [x', y']^T = [[a,b],[c,d]] @ [x,y]^T + [e,f]^T\n"
    "检测到人脸后，终端与画面左上角会实时显示 2×3 矩阵 M=[[a,b,e],[c,d,f]]"
)


def get_face_landmarks(image, timestamp_ms):
    """
    从一帧图像中检测人脸，并提取用于对齐的 6 个关键点。

    参数:
        image: BGR 格式的摄像头帧
        timestamp_ms: 相对起始时间的毫秒数（VIDEO 模式必需，用于帧间跟踪）

    返回:
        shape 为 (6, 2) 的 float32 数组，每行是一个点的 [x, y] 像素坐标；
        未检测到人脸时返回 None。
    """
    # OpenCV 默认 BGR，MediaPipe 需要 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.face_landmarks:
        return None

    h, w = image.shape[:2]
    landmarks = results.face_landmarks[0]  # 只取第一张脸
    # MediaPipe 返回归一化坐标 (0~1)，乘以宽高转为像素坐标
    points = [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in LANDMARK_INDICES]
    return np.array(points, dtype=np.float32)


def overlay_mask(frame, warped_mask):
    """
    将变形后的 mask 叠加到摄像头画面上。

    若 mask 带 Alpha 通道，按透明度逐像素混合，透明区域显示原画面；
    若无 Alpha，则简单加权混合（效果较差，仅作兜底）。
    """
    if warped_mask.shape[2] == 4:
        alpha = warped_mask[:, :, 3:4] / 255.0       # 归一化到 0~1
        foreground = warped_mask[:, :, :3].astype(float)  # BGR 前景
        background = frame.astype(float)                 # BGR 背景（摄像头）
        # 公式：结果 = 背景 × (1 - α) + 前景 × α
        return (background * (1.0 - alpha) + foreground * alpha).astype(np.uint8)

    return cv2.addWeighted(frame, 0.7, warped_mask, 0.3, 0)


def decompose_affine_matrix(matrix):
    """
    将 2×3 仿射矩阵分解为直观参数，便于理解变换含义。

    OpenCV 形式: [x', y']^T = [[a,b],[c,d]] @ [x,y]^T + [e,f]^T
    estimateAffinePartial2D 进一步限制为相似变换（旋转+等比缩放+平移）。
    """
    a, b, e = matrix[0]
    c, d, f = matrix[1]
    scale = float(np.hypot(a, c))
    angle_deg = float(np.degrees(np.arctan2(c, a)))
    return a, b, c, d, e, f, scale, angle_deg


def format_affine_matrix_text(matrix):
    """生成仿射矩阵的可读文本（控制台与画面叠加共用）"""
    a, b, c, d, e, f, scale, angle_deg = decompose_affine_matrix(matrix)
    return [
        "Affine M (2x3):  [x'] = [a b] [x] + [e]",
        "                 [y']   [c d] [y]   [f]",
        f"  a={a:+.4f}  b={b:+.4f}  e={e:+.1f}",
        f"  c={c:+.4f}  d={d:+.4f}  f={f:+.1f}",
        f"  scale={scale:.4f}  rotate={angle_deg:+.2f}deg  tx={e:.1f} ty={f:.1f}",
    ]


def print_affine_matrix(matrix):
    """在终端同一行实时刷新打印仿射矩阵（按 q 退出后会换行）"""
    lines = format_affine_matrix_text(matrix)
    one_line = " | ".join(lines[2:])
    print(f"\r{one_line}", end="", flush=True)


def draw_affine_overlay(frame, matrix):
    """在画面左上角叠加矩阵数值，便于对照观察"""
    lines = format_affine_matrix_text(matrix)
    overlay = frame.copy()
    x0, y0 = 10, 10
    line_h = 22
    box_h = line_h * len(lines) + 16
    box_w = 520
    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        color = (0, 255, 255) if i < 2 else (180, 255, 180)
        cv2.putText(
            frame,
            line,
            (x0, y0 + 16 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def build_mask_points(mask_shape):
    """
    在 mask 图片上定义 6 个控制点的位置（与 LANDMARK_INDICES 对应）。

    优先读取 tools/annotate_mask.py 保存的 mask_points.json；
    若不存在则使用下方默认比例估算。
    """
    h, w = mask_shape[:2]

    if MASK_POINTS_PATH.exists():
        with open(MASK_POINTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        points = np.array(data["points"], dtype=np.float32)
        if points.shape == (6, 2):
            return points
        print("警告：mask_points.json 格式无效，使用默认关键点。")

    return np.array(
        [
            [0.3 * w, 0.35 * h],   # 左眼
            [0.7 * w, 0.35 * h],   # 右眼
            [0.5 * w, 0.5 * h],    # 鼻尖
            [0.35 * w, 0.65 * h],  # 左嘴角
            [0.65 * w, 0.65 * h],  # 右嘴角
            [0.5 * w, 0.85 * h],   # 下巴
        ],
        dtype=np.float32,
    )


# 根据 mask 尺寸预计算控制点，主循环中重复使用
mask_points = build_mask_points(mask.shape)

# ================== 主循环 ==================

cap = cv2.VideoCapture(0)  # 0 = 默认摄像头
if not cap.isOpened():
    print("错误：无法打开摄像头。")
    raise SystemExit(1)

start_time = time.time()  # 记录起始时间，用于生成 VIDEO 模式所需的时间戳

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
title_applied = False

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 读取失败（如摄像头断开）则退出

    # MediaPipe VIDEO 模式要求单调递增的毫秒时间戳
    timestamp_ms = int((time.time() - start_time) * 1000)
    face_points = get_face_landmarks(frame, timestamp_ms)

    if face_points is not None:
        # 根据 mask 控制点与人脸关键点，估算相似变换矩阵（平移+旋转+缩放）
        # 比 getAffineTransform（仅 3 点）更适合 6 点拟合
        matrix, _ = cv2.estimateAffinePartial2D(mask_points, face_points)
        if matrix is not None:
            print_affine_matrix(matrix)

            # 用变换矩阵将 mask 变形到与当前人脸对齐的位置和大小
            warped_mask = cv2.warpAffine(
                mask,
                matrix,
                (frame.shape[1], frame.shape[0]),  # 输出尺寸与摄像头帧一致
                flags=cv2.INTER_LINEAR,             # 双线性插值
                borderMode=cv2.BORDER_TRANSPARENT,    # 空白区域透明
            )
            output = overlay_mask(frame, warped_mask)
            output = draw_affine_overlay(output, matrix)
        else:
            output = frame.copy()  # 变换矩阵计算失败，显示原画面
    else:
        output = frame.copy()  # 未检测到人脸，显示原画面

    cv2.imshow(WINDOW_NAME, output)

    # 首帧显示后窗口才真正创建，此时再设置中文标题
    if not title_applied:
        title_applied = set_opencv_window_title(WINDOW_NAME, WINDOW_TITLE)

    # waitKey(1) 等待 1ms 并处理窗口事件；按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头、关闭检测器、销毁所有 OpenCV 窗口
print()  # 结束 \r 实时打印行
cap.release()
face_landmarker.close()
cv2.destroyAllWindows()
