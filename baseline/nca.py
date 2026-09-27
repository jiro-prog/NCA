"""
Growing Neural Cellular Automata（元祖NCA）の最小実装
1個の「種」セルから、目標の絵（花）を育てるルールを学習する。

使い方:
    pip install torch pillow numpy
    python nca.py              # 学習して、成長の様子を growth.gif に保存
    python nca.py --iters 20   # 動作確認用に短く回す
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------
# ハイパーパラメータ
# ---------------------------------------------------------------
CHANNELS = 16      # 1セルが持つ状態の数（RGBA 4つ + 隠れチャンネル 12個）
HIDDEN = 128       # ルールのネットワークの中間層の幅
FIRE_RATE = 0.5    # 各ステップで更新されるセルの割合（＝サイコロ）
TARGET_SIZE = 40   # 目標の絵の大きさ
PAD = 16           # 絵の周りの余白（成長する空間）
POOL_SIZE = 1024   # サンプルプールの大きさ
BATCH = 8


# ---------------------------------------------------------------
# 目標の絵：花を手続き的に描く（外部画像は使わない）
# RGBは透明度αを掛けた「乗算済みアルファ」形式で持つ
# ---------------------------------------------------------------
def make_target(size=TARGET_SIZE):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((4, size, size), np.float32)
    c = size / 2 - 0.5

    def disc(cx, cy, r, color):
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        a = np.clip(r - d + 0.5, 0, 1)          # 縁をなめらかにした円
        for ch in range(3):                     # 下の色の上に重ねる
            img[ch] = img[ch] * (1 - a) + color[ch] * a
        img[3] = img[3] * (1 - a) + a

    for k in range(6):                          # 花びら6枚
        ang = k * np.pi / 3
        disc(c + 9 * np.cos(ang), c + 9 * np.sin(ang), 7, (0.93, 0.45, 0.6))
    disc(c, c, 6, (0.98, 0.78, 0.2))            # 中心
    t = torch.from_numpy(img)
    return F.pad(t, (PAD, PAD, PAD, PAD))       # (4, H, W)


# ---------------------------------------------------------------
# NCA本体
# ---------------------------------------------------------------
class NCA(nn.Module):
    def __init__(self):
        super().__init__()
        # ② 知覚フィルター（学習しない）：恒等、ソーベルX、ソーベルY
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        sobel_y = sobel_x.T
        k = torch.stack([ident, sobel_x, sobel_y])            # (3, 3, 3)
        k = k.repeat(CHANNELS, 1, 1).unsqueeze(1)              # (48, 1, 3, 3)
        self.register_buffer("kernel", k)

        # ③ ルール：全セル共通の小さなネット（1x1畳み込み＝セルごとのMLP）
        self.rule = nn.Sequential(
            nn.Conv2d(CHANNELS * 3, HIDDEN, 1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False),
        )
        nn.init.zeros_(self.rule[-1].weight)   # 最初は「何もしない」ルール

    def perceive(self, x):
        # 各チャンネルに3種類のフィルターを別々にかける（groups=CHANNELS）
        return F.conv2d(x, self.kernel, padding=1, groups=CHANNELS)

    @staticmethod
    def alive(x):
        # ⑤ 近所3x3のαの最大値が0.1を超えていれば「生きている」
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def fire_mask(self, x):
        # ④ どのセルが今回更新するか。今はサイコロ。
        #    ここを「濃度がしきい値を超えたら発火」に差し替えるのが次の改造ポイント
        b, _, h, w = x.shape
        return (torch.rand(b, 1, h, w, device=x.device) < FIRE_RATE).float()

    def step(self, x):
        pre = self.alive(x)
        dx = self.rule(self.perceive(x))        # ②→③ 変化量を計算
        x = x + dx * self.fire_mask(x)          # ④ 発火したセルだけ更新
        post = self.alive(x)
        return x * (pre & post).float()         # ⑤ 死んだセルは全チャンネル0に

    def forward(self, x, steps):
        for _ in range(steps):
            x = self.step(x)
        return x


def make_seed(h, w, n=1):
    x = torch.zeros(n, CHANNELS, h, w)
    x[:, 3:, h // 2, w // 2] = 1.0              # 中央の1セルだけ、α以降を1に
    return x


def to_rgb(x):
    # 白背景に合成して表示用の画像にする
    rgba = x[:4].clamp(0, 1)
    rgb = 1.0 - rgba[3:4] + rgba[:3]
    return (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------
# 学習
# ---------------------------------------------------------------
def train(iters, device):
    target = make_target().to(device)
    _, h, w = target.shape
    model = NCA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], gamma=0.1)

    # サンプルプール：育ちきった状態から続きを学ばせ、形を「保つ」ことも覚えさせる
    pool = make_seed(h, w, POOL_SIZE)

    for i in range(iters):
        idx = random.sample(range(POOL_SIZE), BATCH)
        x = pool[idx].to(device)

        # いちばん出来の悪いサンプルを種に戻す（ゼロから育てる練習も続ける）
        with torch.no_grad():
            err = ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
            worst = err.argmax()
            x[worst] = make_seed(h, w)[0].to(device)
            # 再生の練習：出来の良いサンプル2つを円形に削る
            for j in err.argsort()[:2]:
                cy, cx = random.randint(PAD, h - PAD), random.randint(PAD, w - PAD)
                yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                hole = ((yy - cy) ** 2 + (xx - cx) ** 2) < 8 ** 2
                x[j][:, hole.to(device)] = 0

        steps = random.randint(64, 96)
        out = model(x, steps)
        loss = ((out[:, :4] - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():            # 勾配を正規化して学習を安定させる
            p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()

        pool[idx] = out.detach().cpu()          # 結果をプールに戻す

        if i % 100 == 0 or i == iters - 1:
            print(f"iter {i:5d}  loss {loss.item():.5f}")

    return model, h, w


def save_growth_gif(model, h, w, device, path="growth.gif", steps=200):
    model.eval()
    x = make_seed(h, w).to(device)
    frames = []
    with torch.no_grad():
        for t in range(steps):
            x = model.step(x)
            if t % 2 == 0:
                img = Image.fromarray(to_rgb(x[0])).resize((w * 4, h * 4), Image.NEAREST)
                frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    m, H, W = train(args.iters, dev)
    torch.save(m.state_dict(), "nca.pt")
    save_growth_gif(m, H, W, dev)
