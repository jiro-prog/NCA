"""
Step 3 のセル（step3_memory/DESIGN.md 9章・10章。2026-10-01 に決まった中身）。ファイル名を cell.py にしたのは、Step 2 の model.py（模擬が `from model import` で読む）と名前がぶつからないようにするため。

Step 2 の凍結したレシピ（DNA なしの B。git のタグ step2-recipe-frozen）を書き換えずに包み、睡眠の段を足す：
  床（sleep=False）   ：凍結したレシピそのもの（step2_attention/model.py の AttentionCell(dna=False)、元の設計の引き金、4歩先の驚き、θ=4）
  眠る段（sleep=True）：床＋睡眠（③）
  記憶の段（sleep=True, memory=True）：眠る段＋保存先 K=4（DESIGN.md 12章。2026-10-02 に決まった中身）

毎ステップ（全セル、固定の安い式。CLAUDE.md の計算の会計）：
  床の1ステップ（凍結したレシピ）
  見直し用の長い痕跡：R_L ← γ_L·R_L ＋ f fᵀ、r_L ← γ_L·r_L ＋ s(t)·f（f = [s(t−1), s(t−2), 1]）。驚いて発火したら0に戻す
  静かさの数え q（最後の驚きの発火からのステップ数）と見直しの回数 k（驚いて発火したら0に戻す）
睡眠の書き込み（数える書き込み。驚きの発火と別枠）：q = N_q·(2^(k+1) − 1) のとき（最後の驚きの発火から100・300・700・1500…ステップ目）
  w ← w ＋ tanh(w_L − w)（w_L は長い痕跡の最小二乗の解）。予測 μ と、いま覚えた4歩先の予測を出し直す
  σ₄² ← σ² ×（1 ＋ α₄² ＋ β₄²）÷（1 ＋ w₁² ＋ w₂²）（新しい係数で。1歩先の外れが入力のノイズだけから来るとしたときの4歩先の外れ）
  k ← k ＋ 1
驚きの発火と睡眠の書き込みは同じステップに重ならない（驚いて発火したら q = 0 なので）。

記憶の段（12章。比べる物差しは D(u) = (u − w)ᵀR(u − w)、基準は 6σ²。σ² はそのセルの v（下限つき））：
  毎ステップ：保存先ごとに、最後に使ってからのステップ数を1つ足す（安い数え）
  思い出し（(お1)、驚いて発火したとき）：最近の痕跡 R・r の最小二乗の解 w_ls に、いちばん合う保存先 u*（D が最小）が D(u*) ≤ 6σ² なら、
    その発火の書き換えの代わりに u* をそのまま書く（同じ1回の出来事。書き込みは1回）。思い出した保存先の数えを0 に戻す
  しまう（睡眠の書き込みのとき、見直しで係数がほとんど動かなかったら：見直しの前の係数 w_前 と長い痕跡の解 w_L で (w_前 − w_L)ᵀR_L(w_前 − w_L) ≤ 6σ²）：
    長い痕跡の上でいちばん合う保存先 u が (u − w)ᵀR_L(u − w) ≤ 6σ² なら u ← u ＋ (w − u)/(n＋1)、n ← n＋1（寄せる）。
    なければ空いている保存先へ、いっぱいなら、いちばん長く使っていないものを入れ替える（忘れ方）

メモリ（1セルあたりの状態の数。DESIGN.md 10.4・12.4）：床は25（係数3、痕跡9、σ・σ₄ 2、濃度・不応期2、最近の値4、覚えている4歩先の予測4、予測1）、
眠る段は36（＋長い痕跡9、静かさの数え1、見直しの回数1）、記憶の段は56（＋保存先4×（係数3、何回しまったか1、最後に使ってから1））。
Step 2 の状態にある隠れ・目覚まし・候補3の窓は、DNA なしの引き金では使わないので数えない。
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
    """睡眠の段と記憶の段の固定の数（DESIGN.md 9.7・12章、台帳 S3-8・S3-10・S3-18〜24）"""
    gamma_L = 0.99          # 見直し用の長い痕跡の減衰（約100ステップ。σ の窓と同じ数）
    n_q = 100               # 最初の見直しまでの静かさ（σ の窓と同じ数）。間隔は2倍ずつ伸ばす（上限なし）
    k_slots = 4             # 保存先の数（12.1。パターンの数3より多い）
    tau = 6.0               # 比べる基準 6σ²（自由な数3つ×2。前の模擬 M1 から同じ数）

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v) and not isinstance(v, classmethod)}


class State3(NamedTuple):
    core: State2            # 床（凍結したレシピ）の状態
    RL: torch.Tensor        # (B,6,H,W) 見直し用の長い痕跡の2次の統計
    rL: torch.Tensor        # (B,3,H,W) 見直し用の長い痕跡の1次の統計
    q: torch.Tensor         # (B,1,H,W) 最後の驚きの発火からのステップ数
    k: torch.Tensor         # (B,1,H,W) 最後の驚きの発火からの見直しの回数
    mslot: torch.Tensor     # (B,K,3,H,W) 保存先の係数（記憶の段だけが使う）
    mn: torch.Tensor        # (B,K,H,W) 保存先ごとの、何回しまったか（0 は空き）
    mage: torch.Tensor      # (B,K,H,W) 保存先ごとの、最後に使ってからのステップ数


def wrap(core: State2) -> State3:
    """床の状態（大人の始め方などで作ったもの）を、睡眠と記憶の状態0で包む"""
    B, _, H, W = core.w.shape
    K = Physio3.k_slots
    z = lambda c: torch.zeros(B, c, H, W, device=core.w.device)
    return State3(core, z(6), z(3), z(1), z(1), torch.zeros(B, K, 3, H, W, device=core.w.device), z(K), z(K))


def init_state(n, h, w, device="cpu") -> State3:
    """予測器0から（育ち）。床の始め方は Step 2 と同じ"""
    return wrap(init_state2(n, h, w, device))


def memory_numbers(sleep: bool, memory: bool = False) -> int:
    """1セルあたりの状態の数（メモリの会計。DESIGN.md 10.4・12.4）"""
    core = 3 + 9 + 2 + 2 + 4 + Physio2.surprise_ahead + 1
    return core + (9 + 1 + 1 if sleep else 0) + (Physio3.k_slots * (3 + 1 + 1) if memory else 0)


def quad(R6, u):
    """uᵀRu（R は上三角の6つ：00,01,02,11,12,22）。u は (B,3,H,W) → (B,H,W)（precheck/storage_check.quad と同じ式）"""
    a, b, c, d, e, f = R6.unbind(1)
    u0, u1, u2 = u.unbind(1)
    return u0 * u0 * a + 2 * u0 * u1 * b + 2 * u0 * u2 * c + u1 * u1 * d + 2 * u1 * u2 * e + u2 * u2 * f


def store(mslot, mn, mage, M, w, RL, sig2):
    """M (B,H,W) のセルに係数 w (B,3,H,W) をしまう（precheck/storage_check.Book.store と同じ計算）。
    戻り値：新しい保存先・数え・最後に使ってから、寄せたセル、新しくしまったセル（空きへ）、入れ替えたセル"""
    K = Physio3.k_slots
    occ = mn > 0
    d = torch.stack([quad(RL, mslot[:, i] - w) for i in range(K)], 1)
    d = torch.where(occ, d, torch.full_like(d, float("inf")))
    d_best, i_best = d.min(1)
    merge = M & (d_best <= Physio3.tau * sig2)
    newm = M & ~merge
    onehot = torch.nn.functional.one_hot(i_best, K).permute(0, 3, 1, 2).bool() & merge.unsqueeze(1)
    n1 = mn + 1
    upd = mslot + (w.unsqueeze(1) - mslot) / n1.unsqueeze(2)
    mslot = torch.where(onehot.unsqueeze(2), upd, mslot)
    mn = torch.where(onehot, n1, mn)
    mage = torch.where(onehot, torch.zeros_like(mage), mage)
    empty = ~occ
    has_empty = empty.any(1)
    ar = torch.arange(K, device=w.device).view(1, K, 1, 1).expand(mn.shape)
    idx_empty = torch.where(empty, ar, torch.full_like(ar, K)).min(1).values
    idx_lru = torch.where(occ, mage, torch.full_like(mage, -1.0)).max(1).indices
    idx = torch.where(has_empty, idx_empty, idx_lru)
    onehot2 = torch.nn.functional.one_hot(idx.clamp(max=K - 1), K).permute(0, 3, 1, 2).bool() & newm.unsqueeze(1)
    mslot = torch.where(onehot2.unsqueeze(2), w.unsqueeze(1).expand_as(mslot), mslot)
    mn = torch.where(onehot2, torch.ones_like(mn), mn)
    mage = torch.where(onehot2, torch.zeros_like(mage), mage)
    return mslot, mn, mage, merge, newm & has_empty, newm & ~has_empty


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
    """Step 3 のセル。sleep=False で床、sleep=True で眠る段、sleep=True・memory=True で記憶の段"""

    def __init__(self, sleep=True, memory=False, device="cpu"):
        if memory and not sleep:
            raise ValueError("記憶の段は眠る段の上に積む（sleep=True が要る）")
        self.sleep = sleep
        self.memory = memory
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
        mslot, mn, mage = s.mslot, s.mn, s.mage
        z0 = torch.zeros_like(fired)
        recalled, compared, merged, stored_new, evicted = z0, torch.zeros_like(q), z0, z0, z0
        if self.memory:
            mage = mage + (mn > 0).float()                                    # 最後に使ってから（毎ステップの安い数え）
            occ = mn > 0
            f0 = fired[:, 0]
            compared = (occ.sum(1).float() * f0.float()).unsqueeze(1)          # 驚きの発火で比べた保存先の数（記録）
            if bool((f0 & occ.any(1)).any()):                                  # 思い出し（(お1)）
                R, r = core.R, core.r
                w_ls = ls_solution(R, r, P.eps)
                D = torch.stack([quad(R, mslot[:, i] - w_ls) for i in range(Physio3.k_slots)], 1)
                D = torch.where(occ, D, torch.full_like(D, float("inf")))
                D_best, i_best = D.min(1)
                sig2 = core.v.clamp(min=P.sigma_min ** 2)[:, 0]
                rc = f0 & (D_best <= Physio3.tau * sig2) & occ.any(1)
                if bool(rc.any()):
                    B, _, H, W = core.w.shape
                    u = torch.gather(mslot, 1, i_best.view(B, 1, 1, H, W).expand(B, 1, 3, H, W)).squeeze(1)
                    core = rewrite(core, u, rc.unsqueeze(1))
                    onehot = torch.nn.functional.one_hot(i_best, Physio3.k_slots).permute(0, 3, 1, 2).bool() & rc.unsqueeze(1)
                    mage = torch.where(onehot, torch.zeros_like(mage), mage)
                recalled = rc.unsqueeze(1)
            sl = sleep_m[:, 0]
            if bool(sl.any()):                                                 # しまう（見直しで落ち着いていたら）
                sig2s = core.v.clamp(min=P.sigma_min ** 2)[:, 0]
                wL = ls_solution(RL, rL, P.eps)
                settled = quad(RL, s.core.w - wL) <= Physio3.tau * sig2s
                mslot, mn, mage, mg, nw, ev = store(mslot, mn, mage, sl & settled, core.w, RL, sig2s)
                merged, stored_new, evicted = mg.unsqueeze(1), nw.unsqueeze(1), ev.unsqueeze(1)
        rec = dict(rec, surprise=fired, sleep=sleep_m, recall=recalled, compared=compared, merged=merged, stored_new=stored_new, evicted=evicted)
        return State3(core, RL, rL, q, k, mslot, mn, mage), rec
