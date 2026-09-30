"""
Step 2 のセル（DESIGN.md 2章・3章。2026-09-29 の決定を反映）。

各セルは自分用の小さな線形予測器（速い重み w = [w₁, w₂, b]）で、次の自分の値を予測する：
    μ(t+1) = w₁·s(t) ＋ w₂·s(t−1) ＋ b

毎ステップ（全セル、固定の安い式）：
    誤差 e = s(t) − μ(t)、z² = e²/σ²
    痕跡（予測器の入力の1次・2次の統計）：R ← γ_R·R ＋ f fᵀ、r ← γ_R·r ＋ s(t)·f（f は μ(t) を出した特徴 [s(t−1), s(t−2), 1]）
    驚きの濃度 c ← 0.9·c ＋ 0.1·z²、自信度 v ← (1−a)·v ＋ a·e²、σ = max(σ_min, √v)（a = 1/σ の窓。窓100）
    引き金：c > θ かつ不応期でない（引いたら c を0に戻し、1ステップ休む）

驚きの信号（2026-09-29、前提の見直しの P3＝候補B。CLAUDE.md「驚きは、少し先の予測の外れで測る」。DESIGN.md 13・14章）：
    Physio2.surprise_ahead = k。k=1 なら上の z²（これまでと1ビットも変わらない）。k>1 なら、k ステップ前にそのときの係数を k 回まわして出した
    k 歩先の予測 μ_k の外れ e_k = s(t) − μ_k(t) で、z_k² = e_k²/σ_k²（σ_k は e_k² の移動平均、窓と下限は σ と同じ）を濃度に溜める。
    k 歩先の予測（係数を k 回まわす読み出し）と σ_k は、全員が毎ステップ更新する安い式。学習の損失は1歩先のまま

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
    sigma_min = 0.02        # σ の下限（センサーのノイズと同じ。2026-09-29 に0.005（S2-7）、2026-09-30 に0.02 に戻した（S2-39））
    sigma_window = 100      # σ の移動平均の窓（案 (B)。2a はこのまま。そうの判断）
    theta = 4.0             # 引き金のしきい値（z² を溜めた濃度）
    c_decay = 0.9           # 濃度の減衰（直近約10ステップ）
    refractory = 1          # 引き金のあと休むステップ数
    gamma_R = 0.9           # 痕跡（R・r）の減衰（直近約10ステップ）
    eps = 1e-3              # 最小二乗の正則化
    h_decay = 0.9           # 書き換えるときの隠れの状態の減衰（|h| ≤ 1/(1−0.9) = 10）
    # 目覚まし（2026-09-29、前提の見直しの候補A。DESIGN 12章）：c ← 0.9c ＋ 0.1z² ＋ d、d = θ·sigmoid(a ＋ d_bias)
    d_bias = 0.0            # Step 1 のレシピと同じ「ずらしなし」（d は θ/2 から始まる）
    alarm_beta = 5.0        # 目覚ましのときの STE の鋭さ（Step 1 と同じ。u = (c − θ)/θ）
    # 驚きの信号（2026-09-29、P3＝候補B）：何歩先の予測の外れで驚くか。1 は1歩先（2026-09-29 まで）。
    # 2026-09-30 から標準は4（標準の世界の切り替え、台帳 S2-39。train.py の --ahead）
    surprise_ahead = 4

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
    d: torch.Tensor      # (B,1,H,W) 目覚まし（目覚ましのときだけ使う。発火したときに DNA が書き、毎ステップ濃度に足される）
    win: torch.Tensor    # (B,1,H,W) 候補3の窓の中の何回目か（0 は窓が閉じている）
    buf: torch.Tensor    # (B,k,H,W) 覚えている k 歩先の予測。buf[:,0] が次のステップの値の予測（いちばん古い）。k = Physio2.surprise_ahead
    vk: torch.Tensor     # (B,1,H,W) k 歩先の誤差の2乗の移動平均（σ_k² のもと）


def init_state(n, h, w, device="cpu"):
    """予測器は0、σ・σ_k は下限、痕跡・濃度・隠れの状態・目覚まし・覚えている予測は0、最近の値は0.5（センサーの真ん中）から始める"""
    z = lambda c: torch.zeros(n, c, h, w, device=device)
    v = torch.full((n, 1, h, w), Physio2.sigma_min ** 2, device=device)
    hist = torch.full((n, N_HIST, h, w), 0.5, device=device)
    return State2(z(3), z(6), z(3), v, z(1), z(1), z(HIDDEN_CH), hist, z(1), z(1), z(1),
                  z(Physio2.surprise_ahead), v.clone())


def ahead(w, x0, x1, k):
    """係数 w = [w₁, w₂, b] を k 回まわした k 歩先の予測（x0 = s(t)、x1 = s(t−1)）。k=1 は1歩先の予測と同じ式"""
    a, b = x0, x1
    for _ in range(k):
        a, b = w[:, 0:1] * a + w[:, 1:2] * b + w[:, 2:3], a
    return a


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
    N_OUT = 2 + 3 + HIDDEN_CH                    # η、g、Δw、Δh（目覚ましのときは ＋1：目覚まし a）

    def __init__(self, dna=True, alarm=False):
        super().__init__()
        self.dna = dna
        self.alarm = alarm                       # 目覚ましの出力を持つか（2a の重みは持たないので、読めるままにする）
        self.rule = nn.Sequential(
            nn.Conv2d(self.N_IN, MLP_HIDDEN, 1),
            nn.ReLU(),
            nn.Conv2d(MLP_HIDDEN, self.N_OUT + (1 if alarm else 0), 1),
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
        """DNA の出力：η（0〜1）、g（0〜1）、Δw（3）、Δh（HIDDEN_CH）、目覚まし a（目覚ましを持つときだけ。なければ None）"""
        if not self.dna:
            B, _, H, W = diff.shape
            one = torch.ones(B, 1, H, W, device=diff.device)
            return one, one, torch.zeros_like(diff), torch.zeros(B, HIDDEN_CH, H, W, device=diff.device), None
        z = self.perceive(self.neighbor_channels(s), diff)
        y = checkpoint(self._net, z, use_reentrant=False) if self.use_checkpoint and z.requires_grad else self._net(z)
        eta = torch.sigmoid(y[:, 0:1])
        g = torch.sigmoid(y[:, 1:2])
        a = y[:, 5 + HIDDEN_CH:6 + HIDDEN_CH] if self.alarm else None
        return eta, g, y[:, 2:5], y[:, 5:5 + HIDDEN_CH], a

    def step(self, s: State2, sense, dense=True, policy=None):
        """1ステップ進める。sense は世界のセンサーの値 (B,1,H,W)。
        発火のしかた policy（dict）。省くと dense=True なら "dense"（2a）、False なら "trigger"（元の設計）：
          dense  ：全員が毎ステップ書き換える（2a）。引き金は数えるだけ
          trigger：元の設計。引き金（c > θ、不応期でない）を引いたセルが1回書き換え、濃度を0に戻して1ステップ休む
          cand3  ：候補3。引き金で窓が開き、濃度がしきい値を下回るまで毎ステップ発火する（上限 K 回）。窓の中では濃度を0に戻さない
          alarm  ：目覚まし（候補A）。c ← 0.9c ＋ 0.1z² ＋ d。d は発火したときに DNA が書く。**このときだけ引き金に STE**
          dice   ：各セルが確率 p で書き換える（対照。gen が要る）／clock：t が k の倍数のとき全員が書き換える（対照。t が要る）
        引き金の数え方（濃度のリセットと不応期）は、候補3 のほかはどれも同じ
        戻り値：新しい状態、記録（e、σ、z²、引き金、書き換え、発火（学習用、目覚ましは STE つき）、発火（0/1）、η、g）"""
        P = Physio2
        pol = policy or {"kind": "dense" if dense else "trigger"}
        kind = pol["kind"]
        f1, f2 = s.hist[:, 0:1], s.hist[:, 1:2]                 # μ(t) を出した特徴 [s(t−1), s(t−2), 1]
        e = sense - s.mu
        e_d = e.detach()
        sigma = torch.clamp(s.v, min=P.sigma_min ** 2).sqrt()   # この回の σ（前のステップまでの誤差から）
        z2 = e_d ** 2 / sigma ** 2
        k = s.buf.shape[1]                                     # 驚きの信号の先（Physio2.surprise_ahead）
        ek = (sense - s.buf[:, 0:1]).detach()                  # k 歩先の外れ（k ステップ前に出した予測）
        sigk = torch.clamp(s.vk, min=P.sigma_min ** 2).sqrt()
        zk2 = ek ** 2 / sigk ** 2
        zt = z2 if k == 1 else zk2                             # 濃度に溜める驚き（k=1 はこれまでと同じ z²）

        # ---- 毎ステップの安い計算（固定の式、全セル。勾配は流さない） ----
        one = torch.ones_like(f1)
        R = P.gamma_R * s.R + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
        r = P.gamma_R * s.r + sense * torch.cat([f1, f2, one], 1)
        c = P.c_decay * s.c + (1 - P.c_decay) * zt
        if kind == "alarm":
            c = c + s.d                                           # 目覚まし：DNA が前の発火で書いた「続ける」合図
        a = 1.0 / P.sigma_window
        v = (1 - a) * s.v + a * e_d ** 2
        vk = (1 - a) * s.vk + a * ek ** 2
        can = s.ref <= 0
        win = s.win
        if kind == "cand3":
            K = pol.get("K", 20)
            trig = (win <= 0) & (c > P.theta) & can               # 窓が開く（驚いた）
            win = torch.where(trig, torch.ones_like(win), torch.where(win > 0, win + 1, win))
            hard = win > 0                                        # 窓の中は毎ステップ発火（最小二乗を解く＝1回と数える）
            done = hard & (c < P.theta)                           # 驚きが収まった：この回は書かずに閉じる
            write = hard & ~done
            close = hard & (done | (win >= K))
            c = torch.where(close, torch.zeros_like(c), c)
            ref = torch.where(close, torch.full_like(s.ref, P.refractory), (s.ref - 1).clamp(min=0))
            win = torch.where(close, torch.zeros_like(win), win)
            fire = write.float()
        else:
            trig = (c > P.theta) & can
            if kind == "alarm":                                   # 目覚ましのときだけ STE（Step 1 と同じ形。u は θ で割る）
                u = (c - P.theta) / P.theta
                soft = (0.5 + 0.5 * P.alarm_beta * u / (1 + P.alarm_beta * u.abs())) * can.float()
                fire = trig.float() + soft - soft.detach()
                hard = trig
            elif kind == "dense":
                hard = torch.ones_like(trig)
                fire = hard.float()
            elif kind == "trigger":
                hard = trig
                fire = hard.float()
            elif kind == "dice":
                hard = torch.rand(trig.shape, device=trig.device, generator=pol["gen"]) < pol["p"]
                fire = hard.float()
            elif kind == "clock":
                tt = pol["t"]
                if torch.is_tensor(tt) and tt.dim() > 0:          # サンプルごとの時刻（学習で時計の時刻を持ち越すとき。2026-09-30、DESIGN.md 17章）
                    hard = ((tt.view(-1, 1, 1, 1) % pol["k"]) == 0).to(trig.device).expand_as(trig)
                else:
                    hard = torch.full_like(trig, pol["t"] % pol["k"] == 0)
                fire = hard.float()
            else:
                raise ValueError(f"発火のしかたが不明：{kind}")
            write = hard
            c = torch.where(trig, torch.zeros_like(c), c)
            ref = torch.where(trig, torch.full_like(s.ref, P.refractory), (s.ref - 1).clamp(min=0))

        # ---- 書き換えるセルだけの高い計算（出力は書き換えるセルの分以外すべて捨てる） ----
        hist = torch.cat([sense, s.hist[:, :-1]], 1)             # 最近の値 [s(t), s(t−1), s(t−2), s(t−3)]
        s_mid = s._replace(R=R, r=r, v=v, c=c, hist=hist)       # 近所が見るのは、このステップの安い計算のあとの値
        w_ls = ls_solution(R, r, P.eps).detach()
        diff = w_ls - s.w
        eta, g, dw, dh, da = self.gates(s_mid, diff)
        wf = fire                                                  # 0/1（目覚ましは順方向 0/1、逆伝播は STE）
        w_new = s.w + wf * (g * s.w + torch.tanh(eta * diff + dw) - s.w)
        h_new = s.h + wf * (P.h_decay * s.h + torch.tanh(dh) - s.h)
        if kind == "alarm":
            if da is None:
                raise ValueError("目覚ましには、目覚ましの出力を持つ DNA が要る（AttentionCell(alarm=True)）")
            d_new = s.d + wf * (P.theta * torch.sigmoid(da + P.d_bias) - s.d)
        else:
            d_new = s.d

        mu = (w_new[:, 0:1] * sense + w_new[:, 1:2] * s.hist[:, 0:1] + w_new[:, 2:3])
        # k 歩先の予測を出して覚える（毎ステップの安い読み出し。驚きにだけ使うので勾配は流さない）
        buf = torch.cat([s.buf[:, 1:], ahead(w_new.detach(), sense, s.hist[:, 0:1], k)], 1)
        new = State2(w_new, R, r, v, c, ref, h_new, hist, mu, d_new, win, buf, vk)
        rec = dict(e=e, sigma=sigma, z2=z2, zk2=zk2, ek=ek, trig=trig, write=write, fire=fire, hard=hard, eta=eta, g=g)
        return new, rec


def nll(e, sigma, detach_sigma=True):
    """学習の損失（DNA を育てる）：(s − μ)²/(2σ²) ＋ log σ。学習では σ を固定値として扱う（σ は DNA が決めない）"""
    sg = sigma.detach() if detach_sigma else sigma
    return e ** 2 / (2 * sg ** 2) + torch.log(sg)


def irreducible(noise, P):
    """2つ前までの線形予測で当てられない誤差（ノイズが入力にも乗る分）"""
    return noise * math.sqrt(2 + 4 * math.cos(2 * math.pi / P) ** 2)
