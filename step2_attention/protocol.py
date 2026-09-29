"""
Step 2 の評価のプロトコル（DESIGN.md 6章）。Step 1 のもの（common/protocol.py）は変えない。

    0〜400：学ぶ（予測器は0、σ は下限から）／400〜800：落ち着く／800：光源の周期を変える（位相はつなげたまま）／800〜1400：学び直す

- 指標はすべて区域ごと（波・ノイズ）。世界は n_worlds 個を並べて回し、セルと世界で平均する
- 周期を変えたあとの指標は、新しい波がその列に届いた時刻（world.column_period で見る）から測る
- 2a（dense=True）は全員が毎ステップ書き換える。引き金は数えるだけ（2b と同じ数え方）
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch

import world as W
from model import init_state, irreducible, nll


@dataclass
class Protocol2:
    total: int = 1400
    settle: tuple = (400, 800)        # 落ち着いた時期
    change_at: int = 800              # この時刻に、新しい周期で進んだ最初の位相が光源に出る
    P1: float = 12.0
    P2: float = 24.0
    n_worlds: int = 4
    relearn: int = 300                # 学び直しの窓（新しい波が届いてから）
    frame_steps: tuple = (0, 5, 20, 100, 399, 799, 810, 850, 900, 950, 1100, 1399)

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def run_protocol2(cell, wcfg, proto, eval_seed, dense=True, device="cpu", frames=False):
    """1回の評価。戻り値：まとめ（dict）、時間ごとの記録（list of dict）、絵のコマ（frames=True のとき）"""
    g = torch.Generator(device=device).manual_seed(eval_seed)
    ws = W.reset(wcfg, proto.P1, n=proto.n_worlds, generator=g, device=device)
    H = Wd = wcfg.size
    st = init_state(proto.n_worlds, H, Wd, device)
    zone = W.zone_mask(wcfg, device)                       # (H,W) ノイズの区域
    wave = ~zone
    T = proto.total
    # セルごとの記録（評価は短いので全部持つ）：誤差の2乗、NLL、z²、引き金、書き換え、σ
    E2 = torch.zeros(T, proto.n_worlds, H, Wd, device=device)
    NL = torch.zeros_like(E2)
    Z2 = torch.zeros_like(E2)
    TR = torch.zeros_like(E2, dtype=torch.bool)
    WR = torch.zeros_like(E2, dtype=torch.bool)
    SG = torch.zeros_like(E2)
    arrival = torch.full((Wd,), -1, dtype=torch.long)
    rows, pics = [], []
    for t in range(T):
        if t == proto.change_at - 1:
            ws = W.set_period(ws, proto.P2)                # 次の advance で新しい周期の位相が光源に出る（時刻 change_at）
        sense = W.sense(ws, wcfg, g)
        st, rec = cell.step(st, sense, dense=dense)
        E2[t] = rec["e"][:, 0] ** 2
        NL[t] = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
        Z2[t] = rec["z2"][:, 0]
        TR[t] = rec["trig"][:, 0]
        WR[t] = rec["write"][:, 0]
        SG[t] = rec["sigma"][:, 0]
        cp = W.column_period(ws, wcfg)[0]                  # (W,) 世界0の列ごとの周期（どの世界も同じ時刻に変える）
        newly = (cp == proto.P2).cpu() & (arrival < 0) & (t >= proto.change_at)
        arrival[newly] = t
        rows.append(dict(t=t, nll_wave=float(NL[t][:, wave].mean()), nll_noise=float(NL[t][:, zone].mean()),
                         z2_wave=float(Z2[t][:, wave].mean()), z2_noise=float(Z2[t][:, zone].mean()),
                         trig_wave=float(TR[t][:, wave].float().mean()), trig_noise=float(TR[t][:, zone].float().mean()),
                         write_wave=float(WR[t][:, wave].float().mean()),
                         sigma_wave=float(SG[t][:, wave].median()), sigma_noise=float(SG[t][:, zone].median()),
                         eta=float(rec["eta"].mean()), g=float(rec["g"].mean())))
        if frames and t in proto.frame_steps:
            pics.append(dict(t=t, sense=sense[0, 0].cpu().numpy(), z=Z2[t][0].cpu().numpy(),
                             sigma=SG[t][0].cpu().numpy(), trig=TR[t][0].cpu().numpy()))
        ws = W.advance(ws)
    return summarize2(E2, NL, Z2, TR, WR, SG, arrival, wave, zone, proto, wcfg), rows, pics


def summarize2(E2, NL, Z2, TR, WR, SG, arrival, wave, zone, proto, wcfg):
    a, b = proto.settle
    out = dict(P1=proto.P1, P2=proto.P2)
    irr = irreducible(wcfg.noise, proto.P1)
    out["irreducible"] = irr
    out["nll_irreducible"] = 0.5 + float(np.log(irr))       # 当てられない誤差の正規分布の NLL（σ が合っているとき）
    for name, m in (("wave", wave), ("noise", zone)):
        out[f"settle_nll_{name}"] = float(NL[a:b][:, :, m].mean())
        out[f"settle_z2_{name}"] = float(Z2[a:b][:, :, m].mean())
        out[f"settle_err_{name}"] = float(E2[a:b][:, :, m].mean().sqrt())
        out[f"settle_trig_{name}"] = float(TR[a:b][:, :, m].float().mean())
        out[f"settle_write_{name}"] = float(WR[a:b][:, :, m].float().mean())
        out[f"sigma_median_{name}"] = float(SG[b - 1][:, m].median())
    out["settle_err_ratio_wave"] = out["settle_err_wave"] / irr
    out["sigma_ratio"] = out["sigma_median_noise"] / out["sigma_median_wave"]
    out["learn_trig_wave"] = float(TR[:a][:, :, wave].float().sum() / (TR.shape[1] * int(wave.sum())))
    # 周期を変えたあと（列ごとに、新しい波が届いた時刻から）
    H, Wd = wave.shape
    cols = []
    for x in range(Wd):
        t0 = int(arrival[x])
        rows_w = wave[:, x]                                  # この列の波のセル
        if t0 < 0 or t0 + proto.relearn > E2.shape[0] or not bool(rows_w.any()):
            continue
        col = lambda A_: A_[..., x][..., rows_w]            # 最後の軸が列 x、その中の波の行
        pre_nll = float(col(NL[a:b]).mean())
        pre_err = float(col(E2[a:b]).mean().sqrt())
        post = slice(t0 + proto.relearn - 50, t0 + proto.relearn)
        tr = col(TR)                                          # (T, n_worlds, 波の行)
        cols.append(dict(
            x=x, arrival=t0,
            react_col=float(tr[t0:t0 + 10].any(0).any(-1).float().mean()),      # この列で10歩以内に引き金（世界で平均）
            react_cell=float(tr[t0:t0 + 10].any(0).float().mean()),
            nll_back=float(col(NL[post]).mean()) <= pre_nll + 0.1,
            err_back=float(col(E2[post]).mean().sqrt()) <= 2 * pre_err,
            gaveup=float((col(SG[t0 + proto.relearn - 1]) > 2 * col(SG[proto.change_at - 1])).float().mean()),
            quiet_after=float(tr[t0 + proto.relearn - 100:t0 + proto.relearn].float().mean()),
            relearn_trig=float(tr[t0:t0 + proto.relearn].float().sum(0).mean())))
    if cols:
        out["react_cols"] = float(np.mean([c["react_col"] >= 0.5 for c in cols]))   # 世界の半分以上で反応した列の割合
        out["react_cell"] = float(np.mean([c["react_cell"] for c in cols]))
        out["nll_back_cols"] = float(np.mean([c["nll_back"] for c in cols]))
        out["err_back_cols"] = float(np.mean([c["err_back"] for c in cols]))
        out["gaveup"] = float(np.mean([c["gaveup"] for c in cols]))
        out["quiet_after_wave"] = float(np.mean([c["quiet_after"] for c in cols]))
        left = [c["relearn_trig"] for c in cols if 1 <= c["x"] <= 8]
        right = [c["relearn_trig"] for c in cols if Wd - 8 <= c["x"] <= Wd - 1]
        out["relearn_trig_left"] = float(np.mean(left)) if left else None
        out["relearn_trig_right"] = float(np.mean(right)) if right else None
        out["left_right_ratio"] = (out["relearn_trig_right"] / out["relearn_trig_left"]
                                   if left and right and out["relearn_trig_left"] > 0 else None)
    out["after_trig_noise"] = float(TR[proto.change_at:][:, :, zone].float().mean())
    out["columns"] = cols
    return out


def judge_2a(s):
    """2a の合格（DESIGN.md 7.1。数字は 2026-09-29 に走らせる前に決めた）：落ち着いた時期（400〜800）だけで判定する"""
    checks = dict(
        sigma_wave=s["sigma_median_wave"] <= 0.015,
        sigma_noise=0.25 <= s["sigma_median_noise"] <= 0.33,
        sigma_ratio=s["sigma_ratio"] >= 5,
        calib_wave=0.5 <= s["settle_z2_wave"] <= 1.5,
        calib_noise=0.5 <= s["settle_z2_noise"] <= 1.5,
        nll_wave=s["settle_nll_wave"] <= s["nll_irreducible"] + 0.2,
    )
    return all(checks.values()), checks


# ---------------------------------------------------------------------------------------------
# 7.5 の評価（2026-09-29、成功条件の決め直し。DESIGN.md 7.5）：育つ時期と大人の時期、対照との比べ方、近所を隠す診断
# ---------------------------------------------------------------------------------------------
import contextlib  # noqa: E402
import math  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from model import N_HIST, AttentionCell, Physio2, ahead  # noqa: E402

PAIRS = ((12.0, 24.0), (12.0, 20.0), (12.0, 36.0), (24.0, 12.0))   # 周期の変化4つ（大人の時期）。12→36 はパッチ S2-8 のあと
PAIRS_ORIGINAL = ((12.0, 24.0), (12.0, 20.0), (24.0, 36.0), (24.0, 12.0))   # パッチを外した元の組（外側のテストは 24→36。DESIGN.md 14章）
CH_HIST0 = 3 + 1 + 1 + 4                                             # 近所のチャンネルのうち、最近の値が始まる番号


def exact_w(P):
    c = math.cos(2 * math.pi / P)
    return (2 * c, -1.0, 1 - c)


def noise_gain(P, k):
    """学び終えた係数を k 回まわしたとき、入力のノイズ n(t)、n(t−1) に掛かる係数 (α_k, β_k)（k 歩先の外れの定常の大きさ σ_n²(1＋α_k²＋β_k²) のため）"""
    w1, w2, _ = exact_w(P)
    al = [0.0, 1.0]      # α_{−1}, α_0
    be = [1.0, 0.0]      # β_{−1}, β_0
    for _ in range(k):
        al.append(w1 * al[-1] + w2 * al[-2])
        be.append(w1 * be[-1] + w2 * be[-2])
    return al[-1], be[-1]


@contextlib.contextmanager
def hide_neighbors():
    """評価のときだけ、近所8か所の最近のセンサーの値を0.5 にする（近所を隠す診断。DESIGN.md 7.5）"""
    orig = AttentionCell.perceive

    def perceive(self, nbr, self_extra):
        B, C, H, Wd = nbr.shape
        p = F.unfold(nbr, 3, padding=1).view(B, C, 9, H, Wd).clone()
        for k in range(N_HIST):
            p[:, CH_HIST0 + k, [0, 1, 2, 3, 5, 6, 7, 8]] = 0.5
        return torch.cat([p.view(B, C * 9, H, Wd), self_extra], 1)
    AttentionCell.perceive = perceive
    try:
        yield
    finally:
        AttentionCell.perceive = orig


@torch.no_grad()
def run_period(cell, policy, wcfg, period, eval_seed=1000, device="cpu", pairs=PAIRS, n_per=4,
               T_grow=400, T_adult=1000, change_at=400, relearn=300):
    """育つ時期（period="grow"：予測器0・σ は下限から、T_grow ステップ。周期は pairs の P1 のまま）か、
    大人の時期（"adult"：学び終えた状態から、change_at で P2 に変えて T_adult まで）を回して、7.5 の指標を返す。
    発火は全セル（ノイズの区域を含む）で数え、質は波の区域の NLL（大人の時期は、新しい波がその列に届いてから relearn ステップの窓）"""
    B = n_per * len(pairs)
    P1 = torch.tensor([a for a, b in pairs for _ in range(n_per)], dtype=torch.float64)
    P2 = torch.tensor([b for a, b in pairs for _ in range(n_per)], dtype=torch.float64)
    g = torch.Generator(device=device).manual_seed(eval_seed)
    g_pol = torch.Generator(device=device).manual_seed(eval_seed + 7)      # サイコロの対照の乱数（世界の乱数と分ける）
    ws = W.reset(wcfg, P1.to(device), n=B, generator=g, device=device)
    H = Wd = wcfg.size
    zone = W.zone_mask(wcfg, device)
    wave = ~zone
    st = init_state(B, H, Wd, device)
    if period == "adult":                                   # 学び終えた状態（表10・13 の診断と同じ始め方）
        frames = []
        for _ in range(N_HIST):
            frames.append(W.sense(ws, wcfg, g))
            ws = W.advance(ws)
        hist = torch.cat(frames[::-1], 1)                   # [s(t), s(t−1), s(t−2), s(t−3)]
        wv = torch.tensor([exact_w(float(p)) for p in P1], device=device, dtype=torch.float32).view(B, 3, 1, 1)
        wv = wv.expand(B, 3, H, Wd).clone()
        irr = torch.tensor([irreducible(wcfg.noise, float(p)) for p in P1], device=device, dtype=torch.float32)
        v = (irr ** 2).clamp(min=Physio_sigma_min() ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
        mu = wv[:, 0:1] * hist[:, 0:1] + wv[:, 1:2] * hist[:, 1:2] + wv[:, 2:3]
        # 驚きの信号の分（k 歩先）：σ_k は学び終えた予測器での k 歩先の外れの定常の大きさ、覚えている k 個の予測は最近の値から学び終えた係数で出す。
        # 覚えている予測 i（0 がいちばん古い）は、時刻 t＋1＋i の値を t＋1＋i−k に出したもの。その時刻の値が最近の値にないときは、t−2 に出した予測で代える
        k = st.buf.shape[1]
        if k == 1:
            vk = v.clone()
        else:
            gk = torch.tensor([1 + sum(x * x for x in noise_gain(float(p), k)) for p in P1], device=device, dtype=torch.float32)
            vk = (wcfg.noise ** 2 * gk).clamp(min=Physio_sigma_min() ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
        hs = [hist[:, i:i + 1] for i in range(N_HIST)]
        slots = []
        for i in range(k):
            j = k - 1 - i
            slots.append(ahead(wv, hs[j], hs[j + 1], k) if j + 1 <= N_HIST - 1 else ahead(wv, hs[N_HIST - 2], hs[N_HIST - 1], i + N_HIST - 1))
        st = st._replace(w=wv, v=v, hist=hist, mu=mu, vk=vk, buf=torch.cat(slots, 1))
    T = T_grow if period == "grow" else T_adult
    NL = torch.zeros(T, B, H, Wd, device=device)
    FI = torch.zeros(T, B, H, Wd, dtype=torch.bool, device=device)
    SG = torch.zeros(T, B, H, Wd, device=device)
    E2 = torch.zeros(T, B, H, Wd, device=device)
    for t in range(T):
        if period == "adult" and t == change_at - 1:
            ws = W.set_period(ws, P2.to(device))
        pol = dict(policy)
        if pol["kind"] == "dice":
            pol["gen"] = g_pol
        if pol["kind"] == "clock":
            pol["t"] = t
        sense = W.sense(ws, wcfg, g)
        st, rec = cell.step(st, sense, policy=pol)
        NL[t] = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
        FI[t] = rec["hard"][:, 0]
        SG[t] = rec["sigma"][:, 0]
        E2[t] = rec["e"][:, 0] ** 2
        ws = W.advance(ws)
    cells = B * H * Wd
    out = dict(period=period, policy=policy["kind"], fires=float(FI.sum()) / cells,
               noise_fire_rate=float(FI[:, :, zone].float().mean()))
    pair_sl = [slice(i * n_per, (i + 1) * n_per) for i in range(len(pairs))]
    pair_name = [f"{a:g}→{b:g}" for a, b in pairs]
    if period == "grow":
        out["quality"] = float(NL[:, :, wave].mean())
        out["wave_fire_rate"] = float(FI[:, :, wave].float().mean())
        out["err_ratio_end"] = float(E2[T - 100:][:, :, wave].mean().sqrt()) / irreducible(wcfg.noise, 12.0)
        out["per_pair"] = {pair_name[i]: dict(fires=float(FI[:, sl].sum()) / (n_per * H * Wd), quality=float(NL[:, sl][:, :, wave].mean()))
                           for i, sl in enumerate(pair_sl)}
        return out
    lag = wcfg.lag
    q, react, back, gave = [], [], [], []
    pp = {n: dict(q=[], react=[], back=[], gave=[]) for n in pair_name}
    for x in range(Wd):
        rows = wave[:, x]
        if not bool(rows.any()):
            continue
        a0 = change_at + x * lag
        if a0 + relearn > T:
            continue
        col = lambda A_: A_[..., x][..., rows]
        q.append(float(col(NL[a0:a0 + relearn]).mean()))
        react.append(float(col(FI[a0:a0 + 10]).any(0).any(-1).float().mean()))
        pre = col(E2[change_at - 200:change_at]).mean(0).sqrt()
        post = col(E2[a0 + relearn - 50:a0 + relearn]).mean(0).sqrt()
        back.append(float((post <= 2 * pre).float().mean()))
        gave.append(float((col(SG[a0 + relearn - 1]) > 2 * col(SG[change_at - 1])).float().mean()))
        for i, sl in enumerate(pair_sl):                    # 組ごと（同じ回し方の中で、組の世界だけを取り出す）
            d = pp[pair_name[i]]
            d["q"].append(float(col(NL[a0:a0 + relearn])[:, sl].mean()))
            d["react"].append(float(col(FI[a0:a0 + 10])[:, sl].any(0).any(-1).float().mean()))
            d["back"].append(float((post[sl] <= 2 * pre[sl]).float().mean()))
            d["gave"].append(float((col(SG[a0 + relearn - 1])[sl] > 2 * col(SG[change_at - 1])[sl]).float().mean()))
    out["per_pair"] = {n: dict(fires=float(FI[:, pair_sl[i]].sum()) / (n_per * H * Wd), quality=float(np.mean(d["q"])),
                               react_cols=float(np.mean(d["react"])), err_back=float(np.mean(d["back"])), gaveup=float(np.mean(d["gave"])))
                       for i, (n, d) in enumerate(pp.items())}
    out["quality"] = float(np.mean(q))
    out["steady_fire_rate_wave"] = float(FI[200:change_at][:, :, wave].float().mean())
    out["steady_fire_rate_noise"] = float(FI[200:change_at][:, :, zone].float().mean())
    out["steady_nll_wave"] = float(NL[200:change_at][:, :, wave].mean())
    out["react_cols"] = float(np.mean(react))
    out["err_back"] = float(np.mean(back))
    out["gaveup"] = float(np.mean(gave))
    out["d_mean_end"] = float(st.d.mean())
    return out


def Physio_sigma_min():
    from model import Physio2
    return Physio2.sigma_min


def envelope(points):
    """対照の点（発火数、NLL）の下の包み：発火を増やしても質が良くならない点を除き、発火数に対して NLL が下がっていく点だけを残す"""
    pts = sorted(points)
    env, best = [], float("inf")
    for f, q in pts:
        if q < best:
            env.append((f, q))
            best = q
    return env


def compare(point, env):
    """7.5 の比べ方。比べる系の点（発火数、NLL）と、対照の下の包み。戻り値：（"上回る"／"負け"／"決められない"、対照の NLL）"""
    f, q = point
    f = max(f, 1e-6)
    if f < env[0][0]:
        return ("上回る" if q < env[0][1] else "決められない"), env[0][1]
    if f > env[-1][0]:
        return ("上回る" if q < env[-1][1] else "負け"), env[-1][1]
    for (f0, q0), (f1, q1) in zip(env[:-1], env[1:]):
        if f0 <= f <= f1:
            if f1 == f0:
                qc = min(q0, q1)
            else:
                t = (math.log(f) - math.log(max(f0, 1e-6))) / (math.log(f1) - math.log(max(f0, 1e-6)))
                qc = q0 + t * (q1 - q0)
            return ("上回る" if q < qc else "負け"), qc
    return ("上回る" if q < env[0][1] else "決められない"), env[0][1]
