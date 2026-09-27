"""
Step 1：濃度発火の花。

元祖NCA（baseline/nca.py）のサイコロを、濃度としきい値による発火に置き換えたもの。
知覚（恒等・ソーベル）、ルールのネット（1×1畳み込み、最終層はゼロ初期化）、生死判定、目標の絵は元祖と同じ。

毎ステップ、全セルがやる安い計算（固定の式。学習しない）:
    濃度 c ← 減衰×c ＋ 目覚まし d ＋ 結合×（近所8つの変化イベントの合計）＋ 誕生の一押し
    c がしきい値以上で、不応期でなければ発火
発火したセルだけがやる高い計算:
    近所を知覚 → ルールのネット → 状態の変化量 dx と、次の目覚まし d を書き込む
    （シミュレーションでは全セルで計算するが、発火したセル以外の出力はすべて捨てる）
発火したあと:
    c を0に戻し、1ステップ休む（不応期）

- 目覚まし d：発火したときにネットが自分に予約する量。毎ステップ濃度に足されるので、
  d が大きいとすぐまた発火し、小さい（1−減衰=0.1 未満）と自分からは二度と起きない
- 変化イベント e：各セルの1ステップの状態の変化の大きさ（チャンネル平均の絶対値、上限1）。
  傷や死による変化も含む。近所はこれを次のステップで受け取る。発火しても変化しなければ、近所は起こさない
- 誕生：生死判定で新しく「生きている」側に入ったセルは、しきい値ぶんの一押しをもらう（最初の1回は必ず発火）
- 結合は、近所1つのイベント（上限1）だけではしきい値に届かない強さにする（興奮性媒質の暴走を防ぐ）
"""
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHANNELS = 16      # 状態（RGBA 4つ＋隠れ12個）。元祖と同じ
HIDDEN = 128


class Physio:
    """発火の仕組みの固定の定数（学習しない）"""
    leak = 0.9          # 濃度が1ステップで残る割合
    theta = 1.0         # しきい値
    kappa = 0.4         # 近所の変化イベントの結合。近所1つだけ（上限1）では 0.4 < しきい値
    e_max = 1.0         # 変化イベントの大きさの上限
    refractory = 1      # 発火後に休むステップ数（発火は最大で2ステップに1回）
    beta = 5.0          # サロゲート勾配の鋭さ
    d_bias = 2.0        # 目覚ましの初期値 sigmoid(0+2)=0.88 → はじめは2ステップに1回くらい発火する

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)
                and not isinstance(v, classmethod)}


class State(NamedTuple):
    x: torch.Tensor     # (B,16,H,W) 状態
    c: torch.Tensor     # (B,1,H,W)  濃度
    d: torch.Tensor     # (B,1,H,W)  目覚まし（発火したときにネットが書き、毎ステップ濃度に足される）
    r: torch.Tensor     # (B,1,H,W)  不応期の残りステップ
    e: torch.Tensor     # (B,1,H,W)  直前のステップの変化の大きさ（次のステップで近所が受け取る）
    was: torch.Tensor   # (B,1,H,W)  直前のステップで生きていたか（誕生の判定用、bool）


def seed_state(h, w, n=1, device="cpu"):
    """中央の1セルだけ、α以降を1にした種。濃度などはすべて0。
    種とその近所は最初のステップで「誕生」するので、必ず発火する"""
    x = torch.zeros(n, CHANNELS, h, w, device=device)
    x[:, 3:, h // 2, w // 2] = 1.0
    z = torch.zeros(n, 1, h, w, device=device)
    return State(x, z.clone(), z.clone(), z.clone(), z.clone(), torch.zeros_like(z, dtype=torch.bool))


def damage_state(s, hole, j=0):
    """サンプル j の hole（(H,W) の bool）のセルを殺す。殺されたセルの変化の大きさは、近所へのイベントになる"""
    s = State(*(t.clone() for t in s))
    change = s.x[j][:, hole].abs().mean(0).clamp(max=Physio.e_max)
    s.e[j, 0][hole] = change
    for t in (s.x, s.c, s.d, s.r):
        t[j][:, hole] = 0
    s.was[j, 0][hole] = False
    return s


class ConcentrationNCA(nn.Module):
    def __init__(self):
        super().__init__()
        n_in = CHANNELS + 1     # 状態＋濃度（濃度は近所から見える）
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        k = torch.stack([ident, sobel_x, sobel_x.T]).repeat(n_in, 1, 1).unsqueeze(1)
        self.register_buffer("kernel", k)                                 # (51, 1, 3, 3)
        nbr8 = torch.ones(3, 3)
        nbr8[1, 1] = 0
        self.register_buffer("nbr8", nbr8.view(1, 1, 3, 3))               # 近所8つの合計
        self.rule = nn.Sequential(
            nn.Conv2d(n_in * 3, HIDDEN, 1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN, CHANNELS + 1, 1, bias=False),               # dx（16）＋ 目覚まし（1）
        )
        nn.init.zeros_(self.rule[-1].weight)    # 最初は「何もしない」ルール（元祖と同じ）
        self.use_checkpoint = False             # True にすると、逆伝播のときにネットを計算し直してVRAMを節約する

    def perceive(self, z):
        return F.conv2d(z, self.kernel, padding=1, groups=z.shape[1])

    def _net(self, z):
        return self.rule(self.perceive(z))

    @staticmethod
    def alive(x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def step(self, s, jitter_p=0.0, generator=None):
        """1ステップ進める。戻り値：新しい状態、発火（学習用のストレートスルー）、発火（0/1）、生きているセル"""
        P = Physio
        x, c, d, r, e, was = s
        pre = self.alive(x)
        pre_f = pre.float()

        # ---- 毎ステップの安い計算（固定の式、全セル） ----
        newborn = (pre & ~was).float()
        nbr = F.conv2d(e, self.nbr8, padding=1)
        c = (P.leak * c + d + P.kappa * nbr + P.theta * newborn) * pre_f
        can = pre & (r <= 0)
        u = c - P.theta
        hard = (u >= 0) & can
        if jitter_p > 0:        # 評価用：環境のゆらぎで、発火が確率 p で1ステップ遅れる
            late = torch.rand(hard.shape, device=hard.device, generator=generator) < jitter_p
            hard = hard & ~late
        hard_f = hard.float()
        soft = (0.5 + 0.5 * P.beta * u / (1 + P.beta * u.abs())) * can.float()
        fire = hard_f + soft - soft.detach()    # 順方向は 0/1、逆伝播は soft の傾き

        # ---- 発火したセルだけの高い計算（出力は発火したセルの分以外すべて捨てる） ----
        z = torch.cat([x, c], 1)
        if self.use_checkpoint and z.requires_grad:
            y = checkpoint(self._net, z, use_reentrant=False)
        else:
            y = self._net(z)
        dx, a = y[:, :CHANNELS], y[:, CHANNELS:]
        x_new = x + dx * fire
        d = d + (torch.sigmoid(a + P.d_bias) - d) * fire

        # ---- 発火したあと・生死判定（固定の式） ----
        c = c * (1 - hard_f)                                                # リセット（勾配は流さない）
        r = torch.where(hard, torch.full_like(r, P.refractory), (r - 1).clamp(min=0))
        live = (pre & self.alive(x_new)).float()
        x_new = x_new * live
        c, d, r = c * live, d * live, r * live
        e = (x_new - x).abs().mean(1, keepdim=True).clamp(max=P.e_max)
        return State(x_new, c, d, r, e, pre), fire, hard, pre

    def forward(self, s, steps, jitter_p=0.0, generator=None):
        """steps ステップ進める。戻り値：最後の状態、サンプルごとの発火数（学習用）、発火数、生きているセル×ステップ数"""
        fire_sum = 0
        n_fire = torch.zeros((), dtype=torch.long, device=s.x.device)
        n_alive = torch.zeros((), dtype=torch.long, device=s.x.device)
        for _ in range(steps):
            s, fire, hard, pre = self.step(s, jitter_p, generator)
            fire_sum = fire_sum + fire.sum(dim=(1, 2, 3))
            n_fire += hard.sum()
            n_alive += pre.sum()
        return s, fire_sum, n_fire, n_alive
