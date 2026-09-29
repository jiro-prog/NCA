"""
Step 2 のセル（DESIGN.md 2章・3章。2026-09-29 の決定を反映）。

各セルは自分用の小さな線形予測器（速い重み w = [w₁, w₂, b]）で、次の自分の値を予測する：
    μ(t+1) = w₁·s(t) ＋ w₂·s(t−1) ＋ b

毎ステップ（全セル、固定の安い式）：
    誤差 e = s(t) − μ(t)、z² = e²/σ²
    痕跡（予測器の入力の1次・2次の統計）：R ← γ_R·R ＋ f fᵀ、r ← γ_R·r ＋ s(t)·f（f は μ(t) を出した特徴 [s(t−1), s(t−2), 1]）
    驚きの濃度 c ← 0.9·c ＋ 0.1·z²、自信度 v ← (1−a)·v ＋ a·e²、σ = max(σ_min, √v)（a = 1/σ の窓。窓100）
    引き金：c > θ かつ不応期でない（引いたら c を0に戻し、1ステップ休む）

書き換えるセル（2a は全員が毎ステップ、2b は引き金を引いたセル。2a でも引き金は数える）：
    最小二乗の解 w_ls = (R ＋ εI)⁻¹ r（3×3 の連立方程式。データだけで決まるので勾配は流さない）
    3×3 の近所の生の値を知覚（近所の係数・log σ・濃度・隠れの状態・最近のセンサーの値。近所の R・r は読まない）＋ 自分の w_ls − w
    DNA（全セル共通の小さな MLP）が η、g、Δw、Δh を出す
    w ← g·w ＋ tanh(η·(w_ls − w) ＋ Δw)、h ← 0.9·h ＋ tanh(Δh)（隠れの状態は減衰つきで |h| ≤ 10）

DNA の最終層はゼロ初期化で、バイアスは η≈0.98、g≈0.998、Δw=0、Δh=0。はじめは「DNA なし」の書き換え（η=1、g=1、Δw=0）とほぼ同じ。
dna=False なら、DNA を回さずに η=1、g=1、Δw=0、Δh=0（DNA なしの対照）。
"""
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HIDDEN_CH = 4       # 隠れの状態のチャンネル数
N_HIST = 4          # 知覚に入れる最近のセンサーの値の数（s(t), s(t−1), s(t−2), s(t−3)）
MLP_HIDDEN = 64


class Physio2:
    """固定の定数（学習しない）"""
    sigma_min = 0.005       # σ の下限（センサーのノイズと同じ。2026-09-29 に0.02から）
    sigma_window = 100      # σ の移動平均の窓（案 (B)。2a はこのまま。そうの判断）
    theta = 4.0             # 引き金のしきい値（z² を溜めた濃度）
    c_decay = 0.9           # 濃度の減衰（直近約10ステップ）
    refractory = 1          # 引き金のあと休むステップ数
    gamma_R = 0.9           # 痕跡（R・r）の減衰（直近約10ステップ）
    eps = 1e-3              # 最小二乗の正則化
    h_decay = 0.9           # 書き換えるときの隠れの状態の減衰（|h| ≤ 1/(1−0.9) = 10）

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)
                and not isinstance(v, classmethod)}


class State2(NamedTuple):
    w: torch.Tensor      # (B,3,H,W) 予測器の係数 [w₁, w₂, b]
    R: torch.Tensor      # (B,6,H,W) 痕跡の2次の統計（3×3 の対称行列の上三角：00,01,02,11,12,22）
    r: torch.Tensor      # (B,3,H,W) 痕跡の1次の統計
    v: torch.Tensor      # (B,1,H,W) 誤差の2乗の移動平均（σ² のもと）
    c: torch.Tensor      # (B,1,H,W) 驚きの濃度
    ref: torch.Tensor    # (B,1,H,W) 不応期の残り
    h: torch.Tensor      # (B,HIDDEN_CH,H,W) 隠れの状態
    hist: torch.Tensor   # (B,N_HIST,H,W) 最近のセンサーの値 [s(t), s(t−1), s(t−2), s(t−3)]（このステップの後）
    mu: torch.Tensor     # (B,1,H,W) 次のステップの予測


def init_state(n, h, w, device="cpu"):
    """予測器は0、σ は下限、痕跡・濃度・隠れの状態は0、最近の値は0.5（センサーの真ん中）から始める"""
    z = lambda c: torch.zeros(n, c, h, w, device=device)
    v = torch.full((n, 1, h, w), Physio2.sigma_min ** 2, device=device)
    hist = torch.full((n, N_HIST, h, w), 0.5, device=device)
    return State2(z(3), z(6), z(3), v, z(1), z(1), z(HIDDEN_CH), hist, z(1))


def _sym(R6):
    """(B,6,H,W) → (B,H,W,3,3)"""
    a, b, c, d, e, f = R6.unbind(1)
    row0 = torch.stack([a, b, c], -1)
    row1 = torch.stack([b, d, e], -1)
    row2 = torch.stack([c, e, f], -1)
    return torch.stack([row0, row1, row2], -2)


def ls_solution(R6, r, eps):
    """最小二乗の解 w_ls = (R＋εI)⁻¹ r。(B,3,H,W)"""
    M = _sym(R6) + eps * torch.eye(3, device=R6.device, dtype=R6.dtype)
    rhs = r.permute(0, 2, 3, 1).unsqueeze(-1)                     # (B,H,W,3,1)
    sol = torch.linalg.solve(M, rhs).squeeze(-1)                  # (B,H,W,3)
    return sol.permute(0, 3, 1, 2)


class AttentionCell(nn.Module):
    N_NBR = 3 + 1 + 1 + HIDDEN_CH + N_HIST     # 近所から読むチャンネル（係数・log σ・濃度・隠れ・最近の値）
    N_IN = N_NBR * 9 + 3                         # 3×3 の生の値 ＋ 自分の w_ls − w
    N_OUT = 2 + 3 + HIDDEN_CH                    # η、g、Δw、Δh

    def __init__(self, dna=True):
        super().__init__()
        self.dna = dna
        self.rule = nn.Sequential(
            nn.Conv2d(self.N_IN, MLP_HIDDEN, 1),
            nn.ReLU(),
            nn.Conv2d(MLP_HIDDEN, self.N_OUT, 1),
        )
        nn.init.zeros_(self.rule[-1].weight)     # はじめは入力によらず一定の出力（DNA なしとほぼ同じ書き換え）
        with torch.no_grad():
            b = self.rule[-1].bias
            b.zero_()
            b[0] = 4.0                           # η = sigmoid(4) ≈ 0.982
            b[1] = 6.0                           # g = sigmoid(6) ≈ 0.9975
        self.use_checkpoint = False

    @staticmethod
    def neighbor_channels(s: State2):
        """近所から読むチャンネル（生の値）：係数 3、log σ 1、濃度 1（log1p）、隠れ 4、最近の値 4。近所の R・r は入れない"""
        sigma = torch.clamp(s.v, min=Physio2.sigma_min ** 2).sqrt()
        return torch.cat([s.w, torch.log(sigma), torch.log1p(s.c), s.h, s.hist], 1)

    def perceive(self, nbr, self_extra):
        """3×3 の近所の生の値を、混ぜずにそのまま並べる（Step 1 のソーベルは使わない）。格子の外は0"""
        B, C, H, W = nbr.shape
        patches = F.unfold(nbr, 3, padding=1).view(B, C * 9, H, W)
        return torch.cat([patches, self_extra], 1)

    def _net(self, z):
        return self.rule(z)

    def gates(self, s: State2, diff):
        """DNA の出力：η（0〜1）、g（0〜1）、Δw（3）、Δh（HIDDEN_CH）"""
        if not self.dna:
            B, _, H, W = diff.shape
            one = torch.ones(B, 1, H, W, device=diff.device)
            return one, one, torch.zeros_like(diff), torch.zeros(B, HIDDEN_CH, H, W, device=diff.device)
        z = self.perceive(self.neighbor_channels(s), diff)
        y = checkpoint(self._net, z, use_reentrant=False) if self.use_checkpoint and z.requires_grad else self._net(z)
        eta = torch.sigmoid(y[:, 0:1])
        g = torch.sigmoid(y[:, 1:2])
        return eta, g, y[:, 2:5], y[:, 5:5 + HIDDEN_CH]

    def step(self, s: State2, sense, dense=True):
        """1ステップ進める。sense は世界のセンサーの値 (B,1,H,W)。
        dense=True（2a）：全員が書き換える。False（2b）：引き金を引いたセルだけが書き換える。どちらでも引き金は同じように数える。
        戻り値：新しい状態、記録（e、σ、z²、引き金、書き換え、η、g）"""
        P = Physio2
        f1, f2 = s.hist[:, 0:1], s.hist[:, 1:2]                 # μ(t) を出した特徴 [s(t−1), s(t−2), 1]
        e = sense - s.mu
        e_d = e.detach()
        sigma = torch.clamp(s.v, min=P.sigma_min ** 2).sqrt()   # この回の σ（前のステップまでの誤差から）
        z2 = e_d ** 2 / sigma ** 2

        # ---- 毎ステップの安い計算（固定の式、全セル。勾配は流さない） ----
        one = torch.ones_like(f1)
        R = P.gamma_R * s.R + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
        r = P.gamma_R * s.r + sense * torch.cat([f1, f2, one], 1)
        c = P.c_decay * s.c + (1 - P.c_decay) * z2
        a = 1.0 / P.sigma_window
        v = (1 - a) * s.v + a * e_d ** 2
        trig = (c > P.theta) & (s.ref <= 0)
        c = torch.where(trig, torch.zeros_like(c), c)
        ref = torch.where(trig, torch.full_like(s.ref, P.refractory), (s.ref - 1).clamp(min=0))
        write = torch.ones_like(trig) if dense else trig

        # ---- 書き換えるセルだけの高い計算（出力は書き換えるセルの分以外すべて捨てる） ----
        hist = torch.cat([sense, s.hist[:, :-1]], 1)             # 最近の値 [s(t), s(t−1), s(t−2), s(t−3)]
        s_mid = s._replace(R=R, r=r, v=v, c=c, hist=hist)       # 近所が見るのは、このステップの安い計算のあとの値
        w_ls = ls_solution(R, r, P.eps).detach()
        diff = w_ls - s.w
        eta, g, dw, dh = self.gates(s_mid, diff)
        wf = write.float()
        w_new = s.w + wf * (g * s.w + torch.tanh(eta * diff + dw) - s.w)
        h_new = s.h + wf * (P.h_decay * s.h + torch.tanh(dh) - s.h)

        mu = (w_new[:, 0:1] * sense + w_new[:, 1:2] * s.hist[:, 0:1] + w_new[:, 2:3])
        new = State2(w_new, R, r, v, c, ref, h_new, hist, mu)
        rec = dict(e=e, sigma=sigma, z2=z2, trig=trig, write=write, eta=eta, g=g)
        return new, rec


def nll(e, sigma, detach_sigma=True):
    """学習の損失（DNA を育てる）：(s − μ)²/(2σ²) ＋ log σ。学習では σ を固定値として扱う（σ は DNA が決めない）"""
    sg = sigma.detach() if detach_sigma else sigma
    return e ** 2 / (2 * sg ** 2) + torch.log(sg)


def irreducible(noise, P):
    """2つ前までの線形予測で当てられない誤差（ノイズが入力にも乗る分）"""
    return noise * math.sqrt(2 + 4 * math.cos(2 * math.pi / P) ** 2)
