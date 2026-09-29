"""
境目を測る実験（BOUNDARY.md。決めごとは 2026-09-29・30 に走らせる前に書いた。台帳 S2-34・S2-39）。

    .venv/Scripts/python.exe step2_attention/boundary.py [--smoke 1] [--device cpu] [--only-tc 125 250] [--only-cv 0 2]

問い：驚き駆動が時計に勝つのは、どんな世界か。変化の間隔の平均 T_c と、間隔のばらつき CV を振った 6×4 の世界で、
DNA なしの驚き駆動（4歩先の驚き、θ×K の格子）と、時計・サイコロの曲線を、同じ世界の実現の上で比べ、勝ち負けの境目を地図にする。

- 世界：標準の世界（32×32、v=0.25、ノイズ0.02、右下のノイズの区域）。光源の周期を、決まった時刻に切り替える（位相はつなげたまま）。
  最初の変化は、書き換えない50ステップのあとの [0, T_c) から一様に。そのあとの間隔はガンマ分布（平均 T_c、CV。CV=0 はちょうど T_c）。
  新しい周期は学習の周期の分布から（見える組に絞らない。予測にほとんど影響しない小さな変化も、世界の一部として入る）
- 始め方：学び終えた状態（はじめの周期の正確な係数、σ は当てられない誤差、σ₄ は定常の大きさ、覚えている予測も埋める）。はじめの50ステップは全員書き換えない
- 長さ：50 ＋ max(4000, 8·T_c)。世界の実現は8つを並べる。センサーの値は世界ごとに1回作って、すべての発火のしかたに配る（組にした比べ）
- 発火のしかた（書き換えはすべて DNA なし）：驚き駆動 θ ∈ {2,3,4,6,8} × K ∈ {1,5,20}（K=1 は元の引き金、K>1 は候補3 の窓）、
  時計 k ∈ {2,…,4096}（書き換えない50ステップのあとから数える）、サイコロ p ∈ {1/2,…,1/4096}
- 物差し：発火の予算 m ＝ 発火率（全セル）× T_c（平均の間隔1つあたりの1セルの発火数）。質＝波の区域の1歩先の NLL の平均（50ステップのあとから）。
  差は変化1回あたりのナット Δ ＝（時計の NLL − 驚き駆動の NLL）× T_c
- 判定（BOUNDARY.md §5）：評価の乱数 3000・3001 の2本とも Δ ＞ 2 なら勝ち、2本とも Δ ＜ −2 なら負け、それ以外は引き分け。
  驚き駆動が m まで発火を減らせないときは「届かない」。m を超えて使えないときは、いちばん多く発火する点で代表させる。見出しは m* = 3、m = 1・10・30 も。
  予算を振った曲線（世界ごとのパレート図）は必ず出す
- 境目を細かくする1回（BOUNDARY.md §7）：m* で隣どうしの判定が分かれたところの真ん中の世界を、最大12 足す
出力：runs/boundary/（summary.json、summary.md、map_m<予算>.png、lines_m3.png、pareto/）。試運転は runs/smoke_boundary/
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from model import N_HIST, Physio2, State2, ahead, init_state, irreducible, ls_solution, nll  # noqa: E402
from protocol import envelope, exact_w, noise_gain  # noqa: E402
from sealed import check_dev_seed  # noqa: E402

TCS = (125, 250, 500, 1000, 2000, 4000)
CVS = (0.0, 0.5, 1.0, 2.0)
THETAS = (2.0, 3.0, 4.0, 6.0, 8.0)
KS = (1, 5, 20)
CLOCK_KS = tuple(2 ** i for i in range(1, 13))
DICE_PS = tuple(1.0 / 2 ** i for i in range(1, 13))
WARMUP = 50
N_WORLDS = 8
EVAL_SEEDS = (3000, 3001)
BUDGETS = (1.0, 3.0, 10.0, 30.0)
M_STAR = 3.0
TIE = 2.0
MAX_REFINE = 12
KIND_SURPRISE, KIND_CLOCK, KIND_DICE = 0, 1, 2


def policies():
    """発火のしかたの一覧（家族、名前、数字）"""
    out = [dict(family="驚き駆動", name=f"θ={th:g} K={K}", kind=KIND_SURPRISE, theta=th, K=K, k=1, p=0.0) for th in THETAS for K in KS]
    out += [dict(family="時計", name=f"k={k}", kind=KIND_CLOCK, theta=Physio2.theta, K=1, k=k, p=0.0) for k in CLOCK_KS]
    out += [dict(family="サイコロ", name=f"p=1/{round(1 / p)}", kind=KIND_DICE, theta=Physio2.theta, K=1, k=1, p=p) for p in DICE_PS]
    return out


def policy_tensors(pols, n_worlds, dev):
    """発火のしかたの数字を、束の要素ごと（発火のしかた×世界の実現。発火のしかたが外側）に並べる"""
    rep = lambda key, dt: torch.tensor([pl[key] for pl in pols for _ in range(n_worlds)], device=dev, dtype=dt).view(-1, 1, 1, 1)
    return dict(kind=rep("kind", torch.long), theta=rep("theta", torch.float32), K=rep("K", torch.float32),
                k=rep("k", torch.long), p=rep("p", torch.float32))


def bstep(s: State2, sense, pol, t_rel, gen, warm=False):
    """境目の実験の1ステップ（DNA なし：η=1、g=1、Δw=0）。model.AttentionCell.step と同じ式で、発火のしかたの数字を要素ごとに持つ。
    驚き駆動は候補3 の窓の決まり（K=1 は元の引き金と同じ）。時計・サイコロは、引き金の数え方（濃度のリセットと不応期）は model と同じで、書き換えは決まった時刻・確率。
    warm=True（はじめの50ステップ）は誰も書き換えない。戻り値：新しい状態、記録（e、σ、発火（0/1））"""
    P = Physio2
    f1, f2 = s.hist[:, 0:1], s.hist[:, 1:2]
    e = sense - s.mu
    sigma = torch.clamp(s.v, min=P.sigma_min ** 2).sqrt()
    z2 = e ** 2 / sigma ** 2
    k = s.buf.shape[1]
    ek = sense - s.buf[:, 0:1]
    sigk = torch.clamp(s.vk, min=P.sigma_min ** 2).sqrt()
    zk2 = ek ** 2 / sigk ** 2
    zt = z2 if k == 1 else zk2
    one = torch.ones_like(f1)
    R = P.gamma_R * s.R + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
    r = P.gamma_R * s.r + sense * torch.cat([f1, f2, one], 1)
    c = P.c_decay * s.c + (1 - P.c_decay) * zt
    a = 1.0 / P.sigma_window
    v = (1 - a) * s.v + a * e ** 2
    vk = (1 - a) * s.vk + a * ek ** 2
    can = s.ref <= 0
    dec = (s.ref - 1).clamp(min=0)
    theta = pol["theta"]
    if warm:
        hard = torch.zeros_like(can)
        write = hard
        win = torch.zeros_like(s.win)
        ref = dec
    else:
        is_s = pol["kind"] == KIND_SURPRISE
        # 驚き駆動（候補3 の窓。model.step の cand3 と同じ）
        win0 = s.win
        trig_s = (win0 <= 0) & (c > theta) & can
        win_s = torch.where(trig_s, torch.ones_like(win0), torch.where(win0 > 0, win0 + 1, win0))
        hard_s = win_s > 0
        done = hard_s & (c < theta)
        write_s = hard_s & ~done
        close = hard_s & (done | (win_s >= pol["K"]))
        c_s = torch.where(close, torch.zeros_like(c), c)
        ref_s = torch.where(close, torch.full_like(s.ref, P.refractory), dec)
        win_s = torch.where(close, torch.zeros_like(win_s), win_s)
        # 時計・サイコロ（model.step の clock・dice と同じ：引き金は数えて濃度を0に戻すが、書き換えは決まった時刻・確率）
        trig_o = (c > theta) & can
        hard_c = ((t_rel % pol["k"]) == 0).expand_as(trig_o)
        hard_d = torch.rand(trig_o.shape, device=trig_o.device, generator=gen) < pol["p"]
        hard_o = torch.where(pol["kind"] == KIND_CLOCK, hard_c, hard_d)
        c_o = torch.where(trig_o, torch.zeros_like(c), c)
        ref_o = torch.where(trig_o, torch.full_like(s.ref, P.refractory), dec)
        hard = torch.where(is_s, hard_s, hard_o)
        write = torch.where(is_s, write_s, hard_o)
        c = torch.where(is_s, c_s, c_o)
        ref = torch.where(is_s, ref_s, ref_o)
        win = torch.where(is_s, win_s, s.win)
    hist = torch.cat([sense, s.hist[:, :-1]], 1)
    w_ls = ls_solution(R, r, P.eps)
    diff = w_ls - s.w
    B_, _, H, Wd = diff.shape
    eta = torch.ones(B_, 1, H, Wd, device=diff.device)
    g = torch.ones(B_, 1, H, Wd, device=diff.device)
    dw = torch.zeros_like(diff)
    wf = write.float()
    w_new = s.w + wf * (g * s.w + torch.tanh(eta * diff + dw) - s.w)
    h_new = s.h + wf * (P.h_decay * s.h + torch.tanh(torch.zeros_like(s.h)) - s.h)
    mu = (w_new[:, 0:1] * sense + w_new[:, 1:2] * s.hist[:, 0:1] + w_new[:, 2:3])
    buf = torch.cat([s.buf[:, 1:], ahead(w_new, sense, s.hist[:, 0:1], k)], 1)
    new = State2(w_new, R, r, v, c, ref, h_new, hist, mu, s.d, win, buf, vk)
    return new, dict(e=e, sigma=sigma, hard=hard)


def schedule(T_c, cv, T_total, rng):
    """1つの世界の変化の時刻（新しい周期の位相が光源に出る時刻）。最初は書き換えない50ステップのあとの [0, T_c) から一様、そのあとはガンマの間隔"""
    times = []
    t = WARMUP + rng.uniform(0, T_c)
    while t < T_total:
        times.append(int(math.floor(t)))
        gap = T_c if cv == 0 else rng.gamma(1 / cv ** 2, T_c * cv ** 2)
        t += max(1.0, gap)
    return sorted(set(times))


@torch.no_grad()
def run_world(T_c, cv, eval_seed, dev, pols, n_worlds=N_WORLDS, length=None):
    """1つの世界の設定（T_c、CV）を、すべての発火のしかた×世界の実現で回し、発火のしかたごとの（発火率、NLL）を返す"""
    wcfg = W.WorldConfig()
    T_total = WARMUP + (length if length is not None else max(4000, 8 * T_c))
    rng = np.random.default_rng(eval_seed * 100003 + int(T_c) * 17 + int(round(cv * 100)))
    gen_cpu = torch.Generator().manual_seed(eval_seed)
    g = torch.Generator(device=dev).manual_seed(eval_seed + 1)
    g_pol = torch.Generator(device=dev).manual_seed(eval_seed + 7)
    P1 = W.sample_periods(n_worlds, gen_cpu)
    scheds = [schedule(T_c, cv, T_total, rng) for _ in range(n_worlds)]
    ws = W.reset(wcfg, P1.to(dev), n=n_worlds, generator=g, device=dev)
    H = Wd = wcfg.size
    n_pol = len(pols)
    B = n_pol * n_worlds
    pt = policy_tensors(pols, n_worlds, dev)
    # 学び終えた状態（run_period の大人の時期と同じ始め方）
    frames = []
    for _ in range(N_HIST):
        frames.append(W.sense(ws, wcfg, g))
        ws = W.advance(ws)
    hist_w = torch.cat(frames[::-1], 1)
    k = Physio2.surprise_ahead
    wv_w = torch.tensor([exact_w(float(p)) for p in P1], device=dev, dtype=torch.float32).view(n_worlds, 3, 1, 1).expand(n_worlds, 3, H, Wd)
    irr = torch.tensor([irreducible(wcfg.noise, float(p)) for p in P1], device=dev, dtype=torch.float32)
    v_w = (irr ** 2).clamp(min=Physio2.sigma_min ** 2).view(n_worlds, 1, 1, 1).expand(n_worlds, 1, H, Wd)
    gk = torch.tensor([1 + sum(x * x for x in noise_gain(float(p), k)) for p in P1], device=dev, dtype=torch.float32)
    vk_w = (wcfg.noise ** 2 * gk).clamp(min=Physio2.sigma_min ** 2).view(n_worlds, 1, 1, 1).expand(n_worlds, 1, H, Wd)
    hs = [hist_w[:, i:i + 1] for i in range(N_HIST)]
    slots = [ahead(wv_w, hs[k - 1 - i], hs[k - i], k) if k - i <= N_HIST - 1 else ahead(wv_w, hs[N_HIST - 2], hs[N_HIST - 1], i + N_HIST - 1)
             for i in range(k)]
    tile = lambda x: x.repeat(n_pol, 1, 1, 1).contiguous()
    mu_w = wv_w[:, 0:1] * hist_w[:, 0:1] + wv_w[:, 1:2] * hist_w[:, 1:2] + wv_w[:, 2:3]
    st = init_state(B, H, Wd, dev)._replace(w=tile(wv_w), v=tile(v_w), hist=tile(hist_w), mu=tile(mu_w), vk=tile(vk_w), buf=tile(torch.cat(slots, 1)))
    wave = ~W.zone_mask(wcfg, dev)
    n_wave = int(wave.sum())
    nll_sum = torch.zeros(B, device=dev, dtype=torch.float64)
    fire_sum = torch.zeros(B, device=dev, dtype=torch.float64)
    next_i = [0] * n_worlds
    n_changes = [0] * n_worlds
    t0 = time.time()
    for t in range(T_total):
        mask = torch.zeros(n_worlds, dtype=torch.bool)
        newP = ws.P.clone().cpu()
        for b in range(n_worlds):                              # 変化の1ステップ前に周期を変える（次の advance で新しい位相が光源に出る）
            if next_i[b] < len(scheds[b]) and scheds[b][next_i[b]] == t + 1:
                mask[b] = True
                newP[b] = W.sample_periods(1, gen_cpu)[0]
                next_i[b] += 1
                if t + 1 >= WARMUP:
                    n_changes[b] += 1
        if bool(mask.any()):
            ws = W.set_period(ws, newP.to(dev), mask=mask.to(dev))
        sense = W.sense(ws, wcfg, g)
        warm = t < WARMUP
        st, rec = bstep(st, sense.repeat(n_pol, 1, 1, 1), pt, torch.tensor(t - WARMUP, device=dev), g_pol, warm=warm)
        if not warm:
            nll_sum += nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0][:, wave].double().sum(1)
            fire_sum += rec["hard"][:, 0].double().sum((1, 2))
        ws = W.advance(ws)
    T_meas = T_total - WARMUP
    fire_rate = (fire_sum / (H * Wd * T_meas)).view(n_pol, n_worlds).mean(1).cpu().numpy()
    nll_mean = (nll_sum / (n_wave * T_meas)).view(n_pol, n_worlds).mean(1).cpu().numpy()
    rows = [dict(family=pl["family"], name=pl["name"], fire_rate=float(fr), m=float(fr * T_c), nll=float(q))
            for pl, fr, q in zip(pols, fire_rate, nll_mean)]
    return dict(T_c=T_c, cv=cv, eval_seed=eval_seed, T_total=T_total, n_changes_mean=float(np.mean(n_changes)),
                seconds=round(time.time() - t0, 1), sec_per_step=round((time.time() - t0) / T_total, 5), policies=rows)


# ------------------------------------------------------------------------------------------------
# 比べ方（BOUNDARY.md §5）
# ------------------------------------------------------------------------------------------------
def env_at(env, m):
    """下の包み env（(m, NLL) を m の小さい順）で、予算 m のときの NLL。
    範囲の中は m の対数で直線に補う。m より多く使えない（最大が m より小さい）ときは最大の点。m まで減らせないときは None（届かない）"""
    if m < env[0][0] * (1 - 1e-9):
        return None
    if m >= env[-1][0]:
        return env[-1][1]
    for (m0, q0), (m1, q1) in zip(env[:-1], env[1:]):
        if m0 <= m <= m1:
            if m1 == m0:
                return min(q0, q1)
            u = (math.log(m) - math.log(m0)) / (math.log(m1) - math.log(m0))
            return q0 + u * (q1 - q0)
    return env[-1][1]


def family_env(res, fam):
    return envelope([(p["m"], p["nll"]) for p in res["policies"] if p["family"] == fam and p["m"] > 0])


def delta(res, m, control="時計"):
    """変化1回あたりのナット Δ ＝（対照の NLL − 驚き駆動の NLL）× T_c。驚き駆動が届かなければ None"""
    qs = env_at(family_env(res, "驚き駆動"), m)
    qc = env_at(family_env(res, control), m)
    if qs is None or qc is None:
        return None
    return (qc - qs) * res["T_c"]


def verdict(r_seeds, m, control="時計"):
    ds = [delta(r, m, control) for r in r_seeds]
    if any(d is None for d in ds):
        return "届かない", ds
    if all(d > TIE for d in ds):
        return "勝ち", ds
    if all(d < -TIE for d in ds):
        return "負け", ds
    return "引き分け", ds


def refine_points(results, m=M_STAR):
    """m* で隣どうしの判定が分かれたところの真ん中の世界（BOUNDARY.md §7。最大12、CV・T_c の順）"""
    v = {(tc, cv): verdict(results[(tc, cv)], m)[0] for tc in TCS for cv in CVS}
    pts = []
    for cv in CVS:
        for a, b in zip(TCS[:-1], TCS[1:]):
            if v[(a, cv)] != v[(b, cv)] and "届かない" not in (v[(a, cv)], v[(b, cv)]):
                pts.append((int(round(math.sqrt(a * b))), cv))
    for tc in TCS:
        for a, b in zip(CVS[:-1], CVS[1:]):
            if v[(tc, a)] != v[(tc, b)] and "届かない" not in (v[(tc, a)], v[(tc, b)]):
                pts.append((tc, (a + b) / 2))
    pts = sorted(set(pts), key=lambda x: (x[1], x[0]))
    return pts[:MAX_REFINE], len(pts)


# ------------------------------------------------------------------------------------------------
# 図
# ------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    return plt


def plot_map(results, m, path, control="時計"):
    plt = _plt()
    from common.viz import INK, MUTED, SERIES, SURFACE
    col = {"勝ち": SERIES[0], "負け": SERIES[1], "引き分け": "#c9c7bf", "届かない": SURFACE}
    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for j, tc in enumerate(TCS):
        for i, cv in enumerate(CVS):
            v, ds = verdict(results[(tc, cv)], m, control)
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=col[v], ec=MUTED if v == "届かない" else "none", lw=0.8))
            txt = v if any(d is None for d in ds) else f"{v}\n{np.mean(ds):+.0f}"
            ax.text(j + 0.48, i + 0.46, txt, ha="center", va="center", fontsize=8, color="white" if v in ("勝ち", "負け") else INK)
    ax.set_xlim(0, len(TCS))
    ax.set_ylim(0, len(CVS))
    ax.set_xticks([j + 0.48 for j in range(len(TCS))], [str(t) for t in TCS])
    ax.set_yticks([i + 0.46 for i in range(len(CVS))], [f"{c:g}" for c in CVS])
    ax.set_xlabel("変化の間隔の平均 T_c（ステップ。右ほどまれ）", fontsize=9, color=INK)
    ax.set_ylabel("不規則さ CV（上ほど固まって来る）", fontsize=9, color=INK)
    ax.set_title(f"驚き駆動 vs {control}　予算 m={m:g}（変化1回あたり1セルの発火）　数字は Δ（変化1回あたりのナット、2本の平均）",
                 loc="left", fontsize=9, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_lines(results, m, path):
    plt = _plt()
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(-TIE, TIE, color=GRID, alpha=0.7, lw=0)
    for i, cv in enumerate(CVS):
        xs, ys = [], []
        for tc in TCS:
            ds = [delta(r, m) for r in results[(tc, cv)]]
            if all(d is not None for d in ds):
                xs.append(tc)
                ys.append(float(np.mean(ds)))
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=SERIES[i], label=f"CV={cv:g}")
    ax.set_xscale("log")
    ax.set_xlabel("変化の間隔の平均 T_c（ステップ）", color=INK_2, fontsize=9)
    ax.set_ylabel("Δ（時計 − 驚き駆動、変化1回あたりのナット。上ほど驚き駆動が良い）", color=INK_2, fontsize=9)
    ax.set_title(f"予算 m={m:g} での差（灰色の帯は引き分けの幅 ±{TIE:g}）", loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_pareto(r_seeds, path):
    plt = _plt()
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE
    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, fam in enumerate(("驚き駆動", "時計", "サイコロ")):
        for j, r in enumerate(r_seeds):
            pts = [(p["m"], p["nll"]) for p in r["policies"] if p["family"] == fam and p["m"] > 0]
            env = envelope(pts)
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, color=SERIES[i], alpha=0.25, linewidths=0)
            ax.plot([p[0] for p in env], [p[1] for p in env], color=SERIES[i], linewidth=1.8 if j == 0 else 1.0,
                    linestyle="-" if j == 0 else "--", label=f"{fam}（乱数{r['eval_seed']}）")
    for m in BUDGETS:
        ax.axvline(m, color=AXIS, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("発火の予算 m（変化1回あたりの1セルの発火。点線は 1・3・10・30）", color=INK_2, fontsize=9)
    ax.set_ylabel("NLL（波の区域、低いほど良い）", color=INK_2, fontsize=9)
    r = r_seeds[0]
    ax.set_title(f"T_c={r['T_c']:g}、CV={r['cv']:g}", loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="1 なら試運転（短く、世界の実現2つ、T_c と CV を2つずつ）。判定には使わない")
    ap.add_argument("--device", default=None)
    ap.add_argument("--refine", type=int, default=1, help="1 なら境目を細かくする1回も回す（BOUNDARY.md §7）")
    args = ap.parse_args()
    for s in EVAL_SEEDS:
        check_dev_seed(s, "評価の乱数")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert Physio2.surprise_ahead == 4 and Physio2.sigma_min == 0.02 and W.WorldConfig().noise == 0.02, "標準の世界で回す（台帳 S2-39）"
    pols = policies()
    tcs, cvs, n_worlds, length = (TCS, CVS, N_WORLDS, None)
    if args.smoke:
        tcs, cvs, n_worlds, length = ((125, 1000), (0.0, 2.0), 2, 600)
    out = HERE / "runs" / ("smoke_boundary" if args.smoke else "boundary")
    (out / "pareto").mkdir(parents=True, exist_ok=True)
    results, log = {}, []
    t_start = time.time()
    for cv in cvs:
        for tc in tcs:
            results[(tc, cv)] = []
            for seed in EVAL_SEEDS:
                r = run_world(tc, cv, seed, dev, pols, n_worlds=n_worlds, length=length)
                results[(tc, cv)].append(r)
                log.append(dict(T_c=tc, cv=cv, seed=seed, seconds=r["seconds"], sec_per_step=r["sec_per_step"], n_changes=r["n_changes_mean"]))
                print(f"T_c={tc} CV={cv:g} 乱数{seed}：{r['seconds']:.0f}秒（1ステップ {r['sec_per_step'] * 1000:.1f}ミリ秒、変化 平均{r['n_changes_mean']:.1f}回）", flush=True)
            plot_pareto(results[(tc, cv)], out / "pareto" / f"tc{tc}_cv{cv:g}.png")
    refined = []
    if args.refine and not args.smoke:
        pts, n_all = refine_points(results)
        for tc, cv in pts:
            results[(tc, cv)] = [run_world(tc, cv, seed, dev, pols) for seed in EVAL_SEEDS]
            plot_pareto(results[(tc, cv)], out / "pareto" / f"tc{tc}_cv{cv:g}.png")
            refined.append(dict(T_c=tc, cv=cv))
            print(f"細かくする：T_c={tc} CV={cv:g} 済み", flush=True)
        if n_all > MAX_REFINE:
            print(f"（分かれたところは {n_all} あったが、決まりどおり最初の {MAX_REFINE} だけ足した）", flush=True)

    summ = dict(settings=dict(TCS=tcs, CVS=cvs, THETAS=THETAS, KS=KS, CLOCK_KS=CLOCK_KS, DICE_PS=DICE_PS, WARMUP=WARMUP, N_WORLDS=n_worlds,
                              EVAL_SEEDS=EVAL_SEEDS, BUDGETS=BUDGETS, M_STAR=M_STAR, TIE=TIE, smoke=bool(args.smoke), length=length),
                refined=refined, log=log, seconds=round(time.time() - t_start, 1), worlds=[])
    for (tc, cv), rs in results.items():
        w = dict(T_c=tc, cv=cv, per_seed=rs, verdict={})
        for ctl in ("時計", "サイコロ"):
            for m in BUDGETS:
                v, ds = verdict(rs, m, ctl)
                w["verdict"][f"{ctl}_m{m:g}"] = dict(verdict=v, deltas=ds)
        env_s = [family_env(r, "驚き駆動") for r in rs]
        w["surprise_m_range"] = [(e[0][0], e[-1][0]) for e in env_s]
        summ["worlds"].append(w)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.smoke:
        for m in BUDGETS:
            plot_map(results, m, out / f"map_m{m:g}.png")
            plot_map(results, m, out / f"map_dice_m{m:g}.png", control="サイコロ")
        plot_lines(results, M_STAR, out / "lines_m3.png")
    L = ["# 境目を測る実験（BOUNDARY.md）" + ("（試運転。判定には使わない）" if args.smoke else ""), "",
         f"世界の設定：T_c {list(tcs)}、CV {list(cvs)}、世界の実現 {n_worlds}、評価の乱数 {list(EVAL_SEEDS)}。かかった時間 {summ['seconds']:.0f}秒", ""]
    for ctl in ("時計", "サイコロ"):
        for m in BUDGETS:
            L += [f"## 驚き駆動 vs {ctl}、予算 m={m:g}" + ("（見出し）" if m == M_STAR and ctl == "時計" else ""), "",
                  "| CV ＼ T_c | " + " | ".join(str(t) for t in tcs) + " |", "|---|" + "---|" * len(tcs)]
            for cv in cvs:
                cells = []
                for tc in tcs:
                    v, ds = verdict(results[(tc, cv)], m, ctl)
                    cells.append(v if any(d is None for d in ds) else f"{v}（{ds[0]:+.1f}／{ds[1]:+.1f}）")
                L.append(f"| {cv:g} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## 驚き駆動が使える予算の範囲（m、下の包みの最小〜最大、乱数ごと）", "", "| T_c | CV | 範囲 |", "|---:|---:|---|"]
    for w in summ["worlds"]:
        L.append(f"| {w['T_c']} | {w['cv']:g} | " + "、".join(f"{a:.2f}〜{b:.1f}" for a, b in w["surprise_m_range"]) + " |")
    if refined:
        L += ["", "## 境目を細かくする1回で足した世界", "", "| T_c | CV | 時計・m=3 の判定 |", "|---:|---:|---|"]
        for rp in refined:
            v, ds = verdict(results[(rp["T_c"], rp["cv"])], M_STAR)
            L.append(f"| {rp['T_c']} | {rp['cv']:g} | {v} {ds} |")
    L += ["", "図：`map_m<予算>.png`（時計）、`map_dice_m<予算>.png`（サイコロ）、`lines_m3.png`、`pareto/tc<T_c>_cv<CV>.png`", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
