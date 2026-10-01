"""
Step 3 の評価（step3_memory/DESIGN.md 7.1・9.2・10.2。開発の模擬 precheck/sleep.py の回し方を、セル（cell.Cell3）を呼ぶ形にしたもの）。
ファイル名を evaluate.py にしたのは、Step 2 の protocol.py（模擬が `from protocol import` で読む）と名前がぶつからないようにするため。

世界：Step 2 の標準の世界（world.WorldConfig）。光源の周期を、並び plan.seq の順に、滞在 plan.dwell ステップごとに切り替える（位相はつなげたまま）。
組 plan.groups の (A, B, C) ごとに世界の実現 plan.n_per 個を並べる。センサーの値は世界の乱数だけで決まる（どのセルでも同じ。組にした比べ）。
始め方：start="grow"（予測器0から最初のパターンで育つ）か "adult"（学び終えた大人の状態。Step 2 の大人の時期と同じ始め方）。
ceiling=True なら天井（波の区域のセルが驚いて発火したら、その列がいま見ている周期の正確な係数を書く。参考で、判定には使わない）。

記録（切り替えごと、セルごと）：
  react     新しい波がその列に届いてから window ステップの中で、一度でも驚いて発火したか
  fires_ph  その滞在の間の驚きの発火の数、sleeps_ph 睡眠の書き込みの数、out_fires 窓の外の驚きの発火の数
  nll_sum   窓の NLL の合計（窓のナットのもと）、num_end/den_end 窓の終わり（届いてから end ステップ）の学び残しのもと
  sig_pre   次の切り替えの直前50ステップの σ₄ の平均
"""
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step2_attention"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cell as C  # noqa: E402
import world as W  # noqa: E402
from model import N_HIST, Physio2, ahead, irreducible, nll  # noqa: E402
from model import init_state as init_state2  # noqa: E402
from protocol import exact_w, noise_gain  # noqa: E402
from sealed import check_dev_seed  # noqa: E402


@dataclass(frozen=True)
class Plan:
    seq: tuple = ("A", "B", "A", "B", "A", "C", "A", "B")
    groups: tuple = ((12.0, 24.0, 8.0), (12.0, 20.0, 8.0))
    n_per: int = 8
    dwell: int = 600
    window: int = 300
    end: tuple = (250, 300)

    def to_dict(self):
        return asdict(self)


def exact_w_t(P):
    """(…) の周期のテンソル → (…, 3) の正確な係数 [2c, −1, 1−c]（protocol.exact_w と同じ式）"""
    c = torch.cos(2 * math.pi / P)
    return torch.stack([2 * c, -torch.ones_like(c), 1 - c], -1)


def sig4_exact(noise, P):
    a, b = noise_gain(P, 4)
    return max(noise * (1 + a * a + b * b) ** 0.5, Physio2.sigma_min)


def adult_core(wcfg, P1, g, dev):
    """学び終えた大人の状態（step2_attention/protocol.run_period の大人の時期と同じ始め方。precheck/headroom.adult_init と同じ）"""
    B = P1.shape[0]
    H = Wd = wcfg.size
    ws = W.reset(wcfg, P1.to(dev), n=B, generator=g, device=dev)
    st = init_state2(B, H, Wd, dev)
    frames = []
    for _ in range(N_HIST):
        frames.append(W.sense(ws, wcfg, g))
        ws = W.advance(ws)
    hist = torch.cat(frames[::-1], 1)
    wv = torch.tensor([exact_w(float(p)) for p in P1], device=dev, dtype=torch.float32).view(B, 3, 1, 1).expand(B, 3, H, Wd).clone()
    irr = torch.tensor([irreducible(wcfg.noise, float(p)) for p in P1], device=dev, dtype=torch.float32)
    v = (irr ** 2).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
    mu = wv[:, 0:1] * hist[:, 0:1] + wv[:, 1:2] * hist[:, 1:2] + wv[:, 2:3]
    k = st.buf.shape[1]
    gk = torch.tensor([1 + sum(x * x for x in noise_gain(float(p), k)) for p in P1], device=dev, dtype=torch.float32)
    vk = (wcfg.noise ** 2 * gk).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
    hs = [hist[:, i:i + 1] for i in range(N_HIST)]
    slots = []
    for i in range(k):
        j = k - 1 - i
        slots.append(ahead(wv, hs[j], hs[j + 1], k) if j + 1 <= N_HIST - 1 else ahead(wv, hs[N_HIST - 2], hs[N_HIST - 1], i + N_HIST - 1))
    st = st._replace(w=wv, v=v, hist=hist, mu=mu, vk=vk, buf=torch.cat(slots, 1))
    return st, ws


@torch.no_grad()
def run(cell, wcfg, seed, start, plan=Plan(), dev="cpu", ceiling=False, sealed=False):
    """1つの世界の実現の束（組×n_per）を、並びの最後まで回して、切り替えごと・セルごとの記録を返す"""
    if not sealed:
        check_dev_seed(seed, "評価の乱数")
    B = plan.n_per * len(plan.groups)
    S_ = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in plan.groups for _ in range(plan.n_per)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in plan.seq]
    T = len(plan.seq) * plan.dwell
    n_sw = len(plan.seq) - 1
    g = torch.Generator(device=dev).manual_seed(seed)
    if start == "adult":
        core, ws = adult_core(wcfg, sched[0], g, dev)
        st = C.wrap(core)
    elif start == "grow":
        ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
        st = C.init_state(B, S_, S_, dev)
    else:
        raise ValueError(start)
    wave = ~W.zone_mask(wcfg, dev)
    xs = torch.arange(S_, device=dev)
    lag = wcfg.lag
    Z = lambda dt=torch.float32: torch.zeros(n_sw, B, S_, S_, device=dev, dtype=dt)
    nll_sum, num_end, den_end, fires_ph, sleeps_ph, out_fires, sig_pre = Z(), Z(), Z(), Z(), Z(), Z(), Z()
    react = Z(torch.bool)
    fires_tot = torch.zeros(B, S_, S_, device=dev)
    sleeps_tot = torch.zeros(B, S_, S_, device=dev)
    t0 = time.time()
    for t in range(T):
        j = t // plan.dwell
        if t % plan.dwell == plan.dwell - 1 and j + 1 < len(plan.seq):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        colP = W.column_period(ws, wcfg)
        core = st.core
        if j >= 1:
            x1, x2 = core.hist[:, 0:1], core.hist[:, 1:2]
            wo = exact_w_t(sched[j - 1].to(dev)).float().view(B, 3, 1, 1)
            wn = exact_w_t(sched[j].to(dev)).float().view(B, 3, 1, 1)
            m_old = wo[:, 0:1] * x1 + wo[:, 1:2] * x2 + wo[:, 2:3]
            m_new = wn[:, 0:1] * x1 + wn[:, 1:2] * x2 + wn[:, 2:3]
            num = (core.mu - m_new) ** 2
            den = (m_old - m_new) ** 2
        st, rec = cell.step(st, sense)
        fired = rec["surprise"]
        if ceiling:
            w_star = exact_w_t(colP).float().permute(0, 2, 1).unsqueeze(2).expand(B, 3, S_, S_)
            st = st._replace(core=C.rewrite(st.core, w_star, fired & wave))
        F = fired[:, 0]
        Sm = rec["sleep"][:, 0]
        fires_tot += F.float()
        sleeps_tot += Sm.float()
        sk = st.core.vk.clamp(min=Physio2.sigma_min ** 2).sqrt()[:, 0]
        if j + 1 < len(plan.seq) and t >= (j + 1) * plan.dwell - 50:
            sig_pre[j] += sk / 50
        if j >= 1:
            sw = j - 1
            NL = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
            rel = (t - (j * plan.dwell + xs * lag)).view(1, 1, S_)
            in_win = (rel >= 0) & (rel < plan.window)
            in_end = (rel >= plan.end[0]) & (rel < plan.end[1])
            nll_sum[sw] += NL * in_win.float()
            num_end[sw] += num[:, 0] * in_end.float()
            den_end[sw] += den[:, 0] * in_end.float()
            fires_ph[sw] += F.float()
            sleeps_ph[sw] += Sm.float()
            react[sw] |= F & in_win
            out_fires[sw] += (F & ~in_win).float()
        ws = W.advance(ws)
    return dict(nll_sum=nll_sum, num_end=num_end, den_end=den_end, fires_ph=fires_ph, sleeps_ph=sleeps_ph, out_fires=out_fires,
                react=react, sig_pre=sig_pre, fires_tot=fires_tot, sleeps_tot=sleeps_tot, sched=sched, wave=wave, elapsed=time.time() - t0)


def summarize(res, wcfg, plan=Plan()):
    """組ごと・切り替えごとの数字（precheck/sleep.py の summarize と同じ物差し）"""
    wave = res["wave"]
    rpc = wave.float().sum(0)
    out = []
    for gi, grp in enumerate(plan.groups):
        sl = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
        rows = []
        for sw in range(len(plan.seq) - 1):
            P_old = float(res["sched"][sw][sl][0])
            P_new = float(res["sched"][sw + 1][sl][0])
            q_col = (res["nll_sum"][sw][sl] * wave).sum((0, 1)) / (rpc * plan.n_per * plan.window)
            ratio = (res["num_end"][sw][sl] / res["den_end"][sw][sl].clamp(min=1e-12))[:, wave]
            rows.append(dict(
                switch=f"{plan.seq[sw]}→{plan.seq[sw + 1]}（{P_old:g}→{P_new:g}）",
                react=float(res["react"][sw][sl][:, wave].float().mean()),
                sig_ratio=float((res["sig_pre"][sw][sl][:, wave] / sig4_exact(wcfg.noise, P_old)).median()),
                fires=float(res["fires_ph"][sw][sl][:, wave].mean()),
                sleeps=float(res["sleeps_ph"][sw][sl][:, wave].mean()),
                restless=float(res["out_fires"][sw][sl][:, wave].mean()) / (max(plan.dwell - plan.window, 1) / 100),
                quality=float(q_col.mean()),
                resid=float(ratio.median())))
        out.append(dict(group=f"A={grp[0]:g}, B={grp[1]:g}, C={grp[2]:g}", rows=rows,
                        fires_life=float(res["fires_tot"][sl].mean()), sleeps_life=float(res["sleeps_tot"][sl].mean())))
    return out


def measure(res, start, gi, plan=Plan()):
    """実数の物差し（DESIGN.md 9.7・10.2）：見逃しの数、書き込み（全セル・波の区域）。precheck/sleep_timing.py の measure と同じ"""
    sl = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
    first = 1 if start == "adult" else 0
    wave = res["wave"]
    r = res["react"][first:, sl][:, :, wave]
    miss = ~r
    w_all = (res["fires_ph"][first:, sl] + res["sleeps_ph"][first:, sl])
    return dict(
        n_miss=int(miss.sum()), ever=int(miss.any(0).sum()), blind_end=int(miss[-1].sum()), n_cells=int(r.shape[1] * r.shape[2]),
        writes_all=float(w_all.mean()), sleeps_all=float(res["sleeps_ph"][first:, sl].mean()),
        writes_wave=float(w_all[:, :, wave].mean()), fires_wave=float(res["fires_ph"][first:, sl][:, :, wave].mean()),
        sleeps_wave=float(res["sleeps_ph"][first:, sl][:, :, wave].mean()))
