"""
Step 3 のセル（step3_memory/DESIGN.md 9章・10章。2026-10-01 に決まった中身）。ファイル名を cell.py にしたのは、Step 2 の model.py（模擬が `from model import` で読む）と名前がぶつからないようにするため。

Step 2 の凍結したレシピ（DNA なしの B。git のタグ step2-recipe-frozen）を書き換えずに包み、睡眠の段を足す：
  床（sleep=False）   ：凍結したレシピそのもの（step2_attention/model.py の AttentionCell(dna=False)、元の設計の引き金、4歩先の驚き、θ=4）
  眠る段（sleep=True）：床＋睡眠（③）。記憶の段は、中身が決まってから足す

毎ステップ（全セル、固定の安い式。CLAUDE.md の計算の会計）：
  床の1ステップ（凍結したレシピ）
  見直し用の長い痕跡：R_L ← γ_L·R_L ＋ f fᵀ、r_L ← γ_L·r_L ＋ s(t)·f（f = [s(t−1), s(t−2), 1]）。驚いて発火したら0に戻す
  静かさの数え q（最後の驚きの発火からのステップ数）と見直しの回数 k（驚いて発火したら0に戻す）
睡眠の書き込み（数える書き込み。驚きの発火と別枠）：q = N_q·(2^(k+1) − 1) のとき（最後の驚きの発火から100・300・700・1500…ステップ目）
  w ← w ＋ tanh(w_L − w)（w_L は長い痕跡の最小二乗の解）。予測 μ と、いま覚えた4歩先の予測を出し直す
  σ₄² ← σ² ×（1 ＋ α₄² ＋ β₄²）÷（1 ＋ w₁² ＋ w₂²）（新しい係数で。1歩先の外れが入力のノイズだけから来るとしたときの4歩先の外れ）
  k ← k ＋ 1
驚きの発火と睡眠の書き込みは同じステップに重ならない（驚いて発火したら q = 0 なので）。

メモリ（1セルあたりの状態の数。DESIGN.md 10.4）：床は25（係数3、痕跡9、σ・σ₄ 2、濃度・不応期2、最近の値4、覚えている4歩先の予測4、予測1）、
眠る段は36（＋長い痕跡9、静かさの数え1、見直しの回数1）。Step 2 の状態にある隠れ・目覚まし・候補3の窓は、DNA なしの引き金では使わないので数えない。
"""
import sys
from pathlib import Path
from typing import NamedTuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step2_attention"))

from model import AttentionCell, Physio2, State2, ahead, ls_solution  # noqa: E402
from model import init_state as init_state2  # noqa: E402


class Physio3:
    """睡眠の段の固定の数（DESIGN.md 9.7、台帳 S3-8・S3-10。2026-10-01 に決まった）"""
    gamma_L = 0.99          # 見直し用の長い痕跡の減衰（約100ステップ。σ の窓と同じ数）
    n_q = 100               # 最初の見直しまでの静かさ（σ の窓と同じ数）。間隔は2倍ずつ伸ばす（上限なし）

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v) and not isinstance(v, classmethod)}


class State3(NamedTuple):
    core: State2            # 床（凍結したレシピ）の状態
    RL: torch.Tensor        # (B,6,H,W) 見直し用の長い痕跡の2次の統計
    rL: torch.Tensor        # (B,3,H,W) 見直し用の長い痕跡の1次の統計
    q: torch.Tensor         # (B,1,H,W) 最後の驚きの発火からのステップ数
    k: torch.Tensor         # (B,1,H,W) 最後の驚きの発火からの見直しの回数


def wrap(core: State2) -> State3:
    """床の状態（大人の始め方などで作ったもの）を、睡眠の状態0で包む"""
    B, _, H, W = core.w.shape
    z = lambda c: torch.zeros(B, c, H, W, device=core.w.device)
    return State3(core, z(6), z(3), z(1), z(1))


def init_state(n, h, w, device="cpu") -> State3:
    """予測器0から（育ち）。床の始め方は Step 2 と同じ"""
    return wrap(init_state2(n, h, w, device))


def memory_numbers(sleep: bool) -> int:
    """1セルあたりの状態の数（メモリの会計。DESIGN.md 10.4）"""
    core = 3 + 9 + 2 + 2 + 4 + Physio2.surprise_ahead + 1
    return core + (9 + 1 + 1 if sleep else 0)


def gain4(w):
    """1 ＋ α₄² ＋ β₄²（係数 w を4回まわしたとき、入力のノイズ n(t)・n(t−1) に掛かる係数。step2_attention/protocol.noise_gain と同じ漸化式）"""
    w1, w2 = w[:, 0:1], w[:, 1:2]
    a_prev, a = torch.zeros_like(w1), torch.ones_like(w1)
    b_prev, b = torch.ones_like(w1), torch.zeros_like(w1)
    for _ in range(4):
        a_prev, a = a, w1 * a + w2 * a_prev
        b_prev, b = b, w1 * b + w2 * b_prev
    return 1 + a * a + b * b


def gain1(w):
    """1 ＋ w₁² ＋ w₂²（1歩先の外れの分散は、入力のノイズの分散の何倍か）"""
    return 1 + w[:, 0:1] ** 2 + w[:, 1:2] ** 2


def rewrite(core: State2, w_new, mask) -> State2:
    """mask のセルの係数を w_new にして、予測 μ と、いま覚えた4歩先の予測（buf の最後）を出し直す（step のあとの状態に使う）"""
    w = torch.where(mask, w_new, core.w)
    x0, x1 = core.hist[:, 0:1], core.hist[:, 1:2]          # s(t)、s(t−1)
    mu = w[:, 0:1] * x0 + w[:, 1:2] * x1 + w[:, 2:3]
    k = core.buf.shape[1]
    buf = torch.cat([core.buf[:, :-1], ahead(w, x0, x1, k)], 1)
    return core._replace(w=w, mu=mu, buf=buf)


class Cell3:
    """Step 3 のセル。sleep=False で床、True で眠る段"""

    def __init__(self, sleep=True, device="cpu"):
        self.sleep = sleep
        self.core = AttentionCell(dna=False).to(device)

    @torch.no_grad()
    def step(self, s: State3, sense):
        """1ステップ進める。戻り値：新しい状態と記録（床の記録に、surprise＝驚きの発火、sleep＝睡眠の書き込み を足したもの）"""
        P = Physio2
        core, rec = self.core.step(s.core, sense, policy={"kind": "trigger"})
        fired = rec["hard"]
        RL, rL, q, k = s.RL, s.rL, s.q, s.k
        sleep_m = torch.zeros_like(fired)
        if self.sleep:
            f1, f2 = core.hist[:, 1:2], core.hist[:, 2:3]                     # このステップの痕跡の入力 [s(t−1), s(t−2), 1]
            one = torch.ones_like(f1)
            RL = Physio3.gamma_L * RL + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
            rL = Physio3.gamma_L * rL + sense * torch.cat([f1, f2, one], 1)
            RL = torch.where(fired, torch.zeros_like(RL), RL)
            rL = torch.where(fired, torch.zeros_like(rL), rL)
            q = torch.where(fired, torch.zeros_like(q), q + 1)
            k = torch.where(fired, torch.zeros_like(k), k)
            sleep_m = q == Physio3.n_q * (torch.pow(2.0, k + 1) - 1)
            if bool(sleep_m.any()):
                wL = ls_solution(RL, rL, P.eps)
                core = rewrite(core, core.w + torch.tanh(wL - core.w), sleep_m)
                v1 = core.v.clamp(min=P.sigma_min ** 2)
                vk_new = (v1 * gain4(core.w) / gain1(core.w)).clamp(min=P.sigma_min ** 2)
                core = core._replace(vk=torch.where(sleep_m, vk_new, core.vk))
            k = k + sleep_m.float()
        rec = dict(rec, surprise=fired, sleep=sleep_m)
        return State3(core, RL, rL, q, k), rec
