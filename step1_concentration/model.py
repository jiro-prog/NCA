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
- 誕生（固定のルール）：生死判定で新しく「生きている」側に入ったセルは、しきい値ぶんの一押しをもらう。
  生まれて最初の発火は「誕生の発火」として別枠で数える
- 結合は、近所1つのイベント（上限1）だけではしきい値に届かない強さにする（興奮性媒質の暴走を防ぐ）
- 近所のイベントのまとめ方（Physio.coupling）：
    sum  … 合計（8つそろうと 0.4×8＝3.2 で、しきい値の3倍。眠ったところで連鎖が起きた）
    mean … 平均（8つそろっても 0.4。起きるには、時間をかけて溜まるか、自分の目覚ましが要る）
    cap  … 合計に上限（coupling_cap）をかける
- 濃度をネットの入力に入れるか（see_c）：既定は入れる（2026-09-28 に一度外したが、外す必要はなかったので戻した）
- タイミングのずれ（評価・学習用の環境のゆらぎ）：発火が確率 p でちょうど1ステップ遅れる。
  遅れた発火は、次のステップで濃度に関係なく必ず起きる（持ち越し）。持ち越しをさらに遅らせることはない
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
    coupling = "sum"    # 近所8つの変化イベントのまとめ方：sum（合計）／mean（平均）／cap（合計に上限）
    coupling_cap = None # coupling="cap" のときの、合計の上限

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)
                and not isinstance(v, classmethod)}


class State(NamedTuple):
    x: torch.Tensor      # (B,16,H,W) 状態
    c: torch.Tensor      # (B,1,H,W)  濃度
    d: torch.Tensor      # (B,1,H,W)  目覚まし（発火したときにネットが書き、毎ステップ濃度に足される）
    r: torch.Tensor      # (B,1,H,W)  不応期の残りステップ
    e: torch.Tensor      # (B,1,H,W)  直前のステップの変化の大きさ（次のステップで近所が受け取る）
    was: torch.Tensor    # (B,1,H,W)  直前のステップで生きていたか（誕生の判定用、bool）
    fresh: torch.Tensor  # (B,1,H,W)  生まれてから、まだ発火していないか（誕生の発火を数える、bool）
    pend: torch.Tensor   # (B,1,H,W)  前のステップで遅らされた発火が、このステップに持ち越されているか（bool）


def seed_state(h, w, n=1, device="cpu"):
    """中央の1セルだけ、α以降を1にした種。濃度などはすべて0。
    種とその近所は最初のステップで「誕生」するので、必ず発火する（ずれで1ステップ遅れることはある）"""
    x = torch.zeros(n, CHANNELS, h, w, device=device)
    x[:, 3:, h // 2, w // 2] = 1.0
    z = torch.zeros(n, 1, h, w, device=device)
    b = torch.zeros_like(z, dtype=torch.bool)
    return State(x, z.clone(), z.clone(), z.clone(), z.clone(), b.clone(), b.clone(), b.clone())


def damage_state(s, hole, j=0):
    """サンプル j の hole（(H,W) の bool）のセルを殺す。殺されたセルの変化の大きさは、近所へのイベントになる"""
    s = State(*(t.clone() for t in s))
    change = s.x[j][:, hole].abs().mean(0).clamp(max=Physio.e_max)
    s.e[j, 0][hole] = change
    for t in (s.x, s.c, s.d, s.r):
        t[j][:, hole] = 0
    for t in (s.was, s.fresh, s.pend):
        t[j, 0][hole] = False
    return s


class ConcentrationNCA(nn.Module):
    def __init__(self, see_c=True):
        super().__init__()
        self.see_c = see_c
        n_in = CHANNELS + (1 if see_c else 0)      # see_c なら状態＋濃度、でなければ状態だけ（元祖と同じ48次元）
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        k = torch.stack([ident, sobel_x, sobel_x.T]).repeat(n_in, 1, 1).unsqueeze(1)
        self.register_buffer("kernel", k)                                 # (n_in*3, 1, 3, 3)
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

    def step(self, s, jitter_p=0.0, generator=None, hardness=1.0):
        """1ステップ進める。
        hardness（学習の初期だけ使う）：1 なら発火は 0/1（ストレートスルー推定）。1 未満なら、順方向の発火を
        hardness×(0/1) ＋ (1−hardness)×(なめらかな量) にして、発火していないセルも少しだけ更新する。
        濃度のリセット・不応期・誕生などの仕組みは、いつも 0/1 で動く。評価はいつも hardness=1
        戻り値：新しい状態、発火（学習用）、発火（0/1）、生きているセル、誕生の発火（0/1）"""
        P = Physio
        x, c, d, r, e, was, fresh, pend = s
        pre = self.alive(x)
        pre_f = pre.float()

        # ---- 毎ステップの安い計算（固定の式、全セル） ----
        newborn = pre & ~was
        fresh = (fresh | newborn) & pre
        nbr = F.conv2d(e, self.nbr8, padding=1)        # 近所8つの変化イベントの合計
        if P.coupling == "mean":
            nbr = nbr / 8
        elif P.coupling == "cap":
            nbr = nbr.clamp(max=P.coupling_cap)
        elif P.coupling != "sum":
            raise ValueError(f"coupling が不明：{P.coupling}")
        c = (P.leak * c + d + P.kappa * nbr + P.theta * newborn.float()) * pre_f
        can = pre & (r <= 0)
        u = c - P.theta
        due = ((u >= 0) | pend) & can                  # しきい値を超えた、または前のステップから持ち越した発火
        if jitter_p > 0:        # 環境のゆらぎ：発火が確率 p でちょうど1ステップ遅れる（持ち越しは遅らせない）
            late = (torch.rand(due.shape, device=due.device, generator=generator) < jitter_p) & ~pend
            hard = due & ~late
            pend = due & late
        else:
            hard = due
            pend = torch.zeros_like(due)
        birth = hard & fresh                            # 生まれて最初の発火
        hard_f = hard.float()
        soft = (0.5 + 0.5 * P.beta * u / (1 + P.beta * u.abs())) * can.float()
        fire = hard_f + soft - soft.detach()            # 順方向は 0/1、逆伝播は soft の傾き
        if hardness < 1.0:                              # 学習の初期：なめらかな量と混ぜる（逆伝播の傾きは同じ）
            fire = hardness * fire + (1.0 - hardness) * soft

        # ---- 発火したセルだけの高い計算（出力は発火したセルの分以外すべて捨てる） ----
        z = torch.cat([x, c], 1) if self.see_c else x
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
        fresh = fresh & ~hard
        live_b = pre & self.alive(x_new)
        live = live_b.float()
        x_new = x_new * live
        c, d, r = c * live, d * live, r * live
        fresh, pend = fresh & live_b, pend & live_b
        e = (x_new - x).abs().mean(1, keepdim=True).clamp(max=P.e_max)
        return State(x_new, c, d, r, e, pre, fresh, pend), fire, hard, pre, birth

    def forward(self, s, steps, jitter_p=0.0, generator=None, hardness=1.0):
        """steps ステップ進める。
        戻り値：最後の状態、サンプルごとの発火数（学習用）、発火数、生きているセル×ステップ数、誕生の発火数"""
        fire_sum = 0
        n_fire = torch.zeros((), dtype=torch.long, device=s.x.device)
        n_alive = torch.zeros((), dtype=torch.long, device=s.x.device)
        n_birth = torch.zeros((), dtype=torch.long, device=s.x.device)
        for _ in range(steps):
            s, fire, hard, pre, birth = self.step(s, jitter_p, generator, hardness)
            fire_sum = fire_sum + fire.sum(dim=(1, 2, 3))
            n_fire += hard.sum()
            n_alive += pre.sum()
            n_birth += birth.sum()
        return s, fire_sum, n_fire, n_alive, n_birth
