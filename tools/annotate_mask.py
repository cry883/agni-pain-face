"""
Mask 关键点可视化与手动标注工具

用法：
  python tools/annotate_mask.py

操作：
  鼠标左键       选中最近的关键点并拖动
  1 ~ 6         切换当前编辑的关键点
  方向键 / WAXD  微调当前点（大写 W/A/X/D 加速移动）
  S             保存到 mask_points.json
  R             恢复默认位置
  Q / Esc       退出
"""

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "assets" / "masks"
MASK_PATH = MASK_DIR / "mask.png"
POINTS_PATH = MASK_DIR / "mask_points.json"
WINDOW_NAME = "Mask Points Editor"

# 与 main.py 中 LANDMARK_INDICES 顺序一致
POINT_LABELS = ["左眼", "右眼", "鼻尖", "左嘴角", "右嘴角", "下巴"]
POINT_COLORS = [
    (0, 255, 255),   # 黄 - 左眼
    (255, 0, 255),   # 洋红 - 右眼
    (0, 165, 255),   # 橙 - 鼻尖
    (0, 255, 0),     # 绿 - 左嘴角
    (255, 255, 0),   # 青 - 右嘴角
    (0, 0, 255),     # 红 - 下巴
]


@lru_cache(maxsize=8)
def _get_font(size: int):
    """加载支持中文的系统字体（Windows 优先微软雅黑）"""
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in font_candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def put_text_cn(canvas_bgr, text, xy, color_bgr, font_size=16):
    """
    在 BGR 图像上绘制中文文本（OpenCV putText 不支持中文，会显示为 ???）。

    xy 为文本左上角像素坐标 (x, y)。
    """
    x, y = xy
    rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def default_points(w, h):
    """默认控制点（与 main.py 中 build_mask_points 一致）"""
    return np.array(
        [
            [0.3 * w, 0.35 * h],
            [0.7 * w, 0.35 * h],
            [0.5 * w, 0.5 * h],
            [0.35 * w, 0.65 * h],
            [0.65 * w, 0.65 * h],
            [0.5 * w, 0.85 * h],
        ],
        dtype=np.float32,
    )


def load_points(w, h):
    """优先从 mask_points.json 加载，否则使用默认值"""
    if POINTS_PATH.exists():
        with open(POINTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        points = np.array(data["points"], dtype=np.float32)
        if points.shape == (6, 2):
            return points
        print("警告：mask_points.json 格式不对，使用默认点。")
    return default_points(w, h)


def save_points(points, w, h):
    """保存像素坐标与比例坐标，供 main.py 读取"""
    ratios = points.copy()
    ratios[:, 0] /= w
    ratios[:, 1] /= h

    data = {
        "mask_size": [w, h],
        "labels": POINT_LABELS,
        "points": points.tolist(),
        "ratios": ratios.tolist(),
    }
    with open(POINTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到 {POINTS_PATH.name}")
    print("像素坐标 (可直接核对):")
    for label, (x, y) in zip(POINT_LABELS, points):
        print(f"  {label}: ({x:.1f}, {y:.1f})")
    print("\n比例坐标 (main.py 会自动从 json 读取，无需手改代码):")
    for label, (rx, ry) in zip(POINT_LABELS, ratios):
        print(f"  {label}: ({rx:.4f}, {ry:.4f})")


def mask_to_bgr(mask):
    """将带 Alpha 的 mask 合成到棋盘格背景上，方便看清透明区域"""
    h, w = mask.shape[:2]
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    block = 16
    for y in range(0, h, block):
        for x in range(0, w, block):
            color = 180 if ((x // block) + (y // block)) % 2 == 0 else 120
            checker[y : y + block, x : x + block] = color

    if mask.shape[2] == 4:
        alpha = mask[:, :, 3:4].astype(float) / 255.0
        bgr = mask[:, :, :3].astype(float)
        return (checker * (1.0 - alpha) + bgr * alpha).astype(np.uint8)
    return mask[:, :, :3].copy()


def draw_overlay(base_bgr, points, selected_idx, dragging_idx):
    """在 mask 上绘制关键点、编号和辅助线"""
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]

    # 辅助线分三段绘制，避免下巴只连一侧
    guide_lines = [
        [0, 2, 1],   # 左眼 - 鼻尖 - 右眼
        [3, 4],      # 左嘴角 - 右嘴角
        [3, 5, 4],   # 左嘴角 - 下巴 - 右嘴角
    ]
    for indices in guide_lines:
        poly = points[indices].astype(np.int32)
        cv2.polylines(canvas, [poly], False, (80, 80, 80), 1, cv2.LINE_AA)

    for i, (point, label, color) in enumerate(zip(points, POINT_LABELS, POINT_COLORS)):
        x, y = int(point[0]), int(point[1])
        is_active = i == selected_idx or i == dragging_idx
        radius = 10 if is_active else 7
        thickness = 3 if is_active else 2

        cv2.circle(canvas, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        canvas = put_text_cn(
            canvas,
            f"{i + 1}.{label}",
            (x + 12, y - 22),
            color,
            font_size=15,
        )

    # 顶部操作提示
    hint = "拖动/方向键/WAXD | 1-6切换 | S保存 | R重置 | Q退出"
    cv2.rectangle(canvas, (0, 0), (w, 32), (30, 30, 30), -1)
    canvas = put_text_cn(canvas, hint, (8, 6), (220, 220, 220), font_size=14)

    active = dragging_idx if dragging_idx >= 0 else selected_idx
    info = f"当前: {active + 1}.{POINT_LABELS[active]}  ({points[active, 0]:.1f}, {points[active, 1]:.1f})"
    cv2.rectangle(canvas, (0, h - 28), (w, h), (30, 30, 30), -1)
    canvas = put_text_cn(canvas, info, (8, h - 24), POINT_COLORS[active], font_size=14)

    return canvas


class Editor:
    def __init__(self, mask, points):
        self.base_bgr = mask_to_bgr(mask)
        self.h, self.w = mask.shape[:2]
        self.points = points.copy()
        self.selected_idx = 0
        self.dragging_idx = -1
        self.mouse_down = False

    def nearest_point(self, x, y, max_dist=20):
        dists = np.linalg.norm(self.points - np.array([x, y], dtype=np.float32), axis=1)
        idx = int(np.argmin(dists))
        return idx if dists[idx] <= max_dist else -1

    def clamp_point(self, idx):
        self.points[idx, 0] = np.clip(self.points[idx, 0], 0, self.w - 1)
        self.points[idx, 1] = np.clip(self.points[idx, 1], 0, self.h - 1)

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.nearest_point(x, y)
            if idx >= 0:
                self.selected_idx = idx
                self.dragging_idx = idx
                self.mouse_down = True
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down and self.dragging_idx >= 0:
            self.points[self.dragging_idx] = [x, y]
            self.clamp_point(self.dragging_idx)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            self.dragging_idx = -1

    def nudge(self, dx, dy):
        self.points[self.selected_idx, 0] += dx
        self.points[self.selected_idx, 1] += dy
        self.clamp_point(self.selected_idx)

    def render(self):
        return draw_overlay(self.base_bgr, self.points, self.selected_idx, self.dragging_idx)


# 各平台 OpenCV 方向键码（waitKeyEx 返回值）
ARROW_DELTAS = {
    2424832: (-1, 0),   # Left  (Windows)
    2490368: (0, -1),   # Up
    2555904: (1, 0),    # Right
    2621440: (0, 1),    # Down
    65361: (-1, 0),     # Left  (Linux/macOS)
    65362: (0, -1),     # Up
    65363: (1, 0),      # Right
    65364: (0, 1),      # Down
}


def main():
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"错误：找不到 {MASK_PATH}")
        raise SystemExit(1)

    h, w = mask.shape[:2]
    print(f"Mask 尺寸: {w} x {h} (宽 x 高)")
    if POINTS_PATH.exists():
        print(f"已加载 {POINTS_PATH.name}")
    else:
        print("使用默认关键点，调整后按 S 保存")

    editor = Editor(mask, load_points(w, h))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, editor.on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, editor.render())
        key = cv2.waitKeyEx(20)

        if key == -1:
            continue

        low = key & 0xFF
        fast = key != low  # 扩展键（方向键等）或按住 Shift 时加速

        if low in (ord("q"), ord("Q"), 27):
            break
        if low in (ord("s"), ord("S")):
            save_points(editor.points, w, h)
        if low in (ord("r"), ord("R")):
            editor.points = default_points(w, h)
            print("已恢复默认关键点")
        if ord("1") <= low <= ord("6"):
            editor.selected_idx = low - ord("1")

        step = 5 if fast else 1
        moved = False

        if key in ARROW_DELTAS:
            dx, dy = ARROW_DELTAS[key]
            editor.nudge(dx * step, dy * step)
            moved = True
        elif low == ord("w"):
            editor.nudge(0, -step)
            moved = True
        elif low == ord("a"):
            editor.nudge(-step, 0)
            moved = True
        elif low == ord("d"):
            editor.nudge(step, 0)
            moved = True
        elif low == ord("x"):
            editor.nudge(0, step)
            moved = True

        if moved:
            pass  # 下一帧自动重绘

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
