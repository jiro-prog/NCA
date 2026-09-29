"""
目標の絵のほかの形（2026-09-29、そうの判断：Step 1 のレシピが花に特化していないかを確かめる）。

花は元祖（baseline/nca.py の make_target）をそのまま使い、ここには置かない（元祖は改造しない）。
ここに置く形も、花と同じく手続き的に描く（外部画像は使わない）。大きさは花と同じ 40×40 に余白16、乗算済みアルファ。

- gecko（トカゲ）：横向きの胴、頭、4本の脚、曲がった尾、目、背中の斑点。
  花（6回対称のかたまり）と違い、左右も上下も対称でなく、細い脚と尾がある（元祖の論文の代表的な目標もトカゲ）
"""
import numpy as np
import torch
import torch.nn.functional as F

TARGET_SIZE = 40
PAD = 16

GREEN = (0.30, 0.68, 0.34)
SPOT = (0.96, 0.82, 0.25)
EYE = (0.08, 0.08, 0.10)


def _painter(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((4, size, size), np.float32)

    def paint(a, color):
        for ch in range(3):                     # 下の色の上に重ねる（花と同じ重ね方）
            img[ch] = img[ch] * (1 - a) + color[ch] * a
        img[3] = img[3] * (1 - a) + a

    def disc(cx, cy, r, color):
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        paint(np.clip(r - d + 0.5, 0, 1), color)   # 縁をなめらかにした円（花と同じ）

    def ellipse(cx, cy, ax, ay, color):
        d = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
        paint(np.clip((1 - d) * min(ax, ay) + 0.5, 0, 1), color)

    def limb(p0, p1, r, color, n=12):
        """p0 から p1 へ、同じ太さの円を並べた線"""
        for t in np.linspace(0, 1, n):
            disc(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, r, color)

    return img, disc, ellipse, limb


def make_gecko(size=TARGET_SIZE, pad=PAD):
    img, disc, ellipse, limb = _painter(size)
    c = size / 2 - 0.5
    # 脚（胴より先に描き、胴を上に重ねる）：前脚は x=c−5、後ろ脚は x=c+5 の胴の縁から斜めに出る
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        x0 = c + 5 * sx
        hip = (x0, c + 3 * sy)
        knee = (x0 + 3 * sx, c + 8 * sy)
        foot = (x0 + 6 * sx, c + 10 * sy)
        limb(hip, knee, 1.8, GREEN)
        limb(knee, foot, 1.6, GREEN)
        disc(foot[0], foot[1], 2.2, GREEN)
    # 尾：胴の右端から、右下へ曲がりながら細くなる（2次のベジエ曲線に沿って円を並べる）
    p0, p1, p2 = np.array([c + 9, c + 0.5]), np.array([c + 16, c + 1.5]), np.array([c + 17.5, c + 9])
    for t in np.linspace(0, 1, 28):
        p = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        disc(p[0], p[1], 3.0 - 1.8 * t, GREEN)
    ellipse(c + 1, c, 10.5, 5.5, GREEN)          # 胴（横長）
    disc(c - 11, c - 0.5, 4.6, GREEN)             # 頭
    for dx in (-3.5, 0.5, 4.5):                   # 背中の斑点
        disc(c + dx, c, 1.6, SPOT)
    for sy in (-1, 1):                            # 目
        disc(c - 12.5, c - 0.5 + 2.2 * sy, 1.1, EYE)
    t = torch.from_numpy(img)
    return F.pad(t, (pad, pad, pad, pad))         # (4, H, W)


TARGETS = {"gecko": make_gecko}


def target_fn(name, flower_fn):
    """名前から目標の絵を作る関数を返す。flower は元祖の make_target をそのまま使う"""
    if name == "flower":
        return flower_fn
    if name in TARGETS:
        return TARGETS[name]
    raise ValueError(f"目標の絵の名前が不明：{name}（flower か {', '.join(TARGETS)}）")
