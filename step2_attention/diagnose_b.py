"""
仮説「遠くを予測した驚きは、学び終わりの合図にもなる」の確かめ（DESIGN.md 15章。読み方は 2026-09-29 に走らせる前に決めた）。

    .venv/Scripts/python.exe step2_attention/diagnose_b.py [--runs runs/2b_B_seed0 runs/2b_B_seed1] [--device cpu]

大人の時期（学び終えた状態から、400 で周期を変えて1000 まで。組は 12→24・12→20・24→36・24→12、評価の乱数1000）で：
1. 物差しの空の確かめ：一度も書き換えない（サイコロ p=0）ときの、あきらめ・戻り・NLL（ノイズ0.02 と0.005）
2. 学び残し：同じセンサーの値の上で、(今の予測器の1歩先の予測 − 新しい周期の正確な予測器の予測)² を、(古い周期の正確な予測器 − 新しい周期の正確な予測器)² で割る
   （どちらも同じ区間で平均してから割る）。止まったとき（窓の中の最後の発火のあと50ステップ）と、窓の終わり（届いてから250〜300）。床は密の窓の終わり
3. 補助：学び残し（直近10ステップ）で区間に分けた z₄² と z² の平均（B の回し方で）
比べるもの：DNA なしの B（4歩先）、1歩先の驚き（元の設計）、密、書き換えない。ノイズ0.02 と0.005（σ の下限はノイズと同じ）。ノイズ0.02 では学んだ B の DNA も
判定（15章）：決め手はノイズ0.005。組 12→24・12→20・24→12 のすべてで、B の止まったときの学び残し（中央値）が 床＋0.1 以下、
  かつ 1歩先の驚きのそれが 床＋0.3 以上 → 仮説を支持。B がどれかの組で 床＋0.3 以上 → 仮説に反する。それ以外はどちらとも言えない
出力：runs/diagnose_b/（summary.md、summary.json）
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from model import N_HIST, AttentionCell, Physio2, ahead, init_state, irreducible, nll  # noqa: E402
from protocol import PAIRS_ORIGINAL, exact_w, noise_gain, run_period  # noqa: E402

JUDGE = ("12→24", "12→20", "24→12")
BINS = (0.0, 0.1, 0.2, 0.4, 0.7, float("inf"))
POLICIES = {                                     # 名前：（驚きの先、発火のしかた）
    "B（4歩先、DNA なし）": (4, {"kind": "trigger"}),
    "1歩先の驚き（DNA なし）": (1, {"kind": "trigger"}),
    "密（DNA なし）": (1, {"kind": "dense"}),
    "書き換えない": (1, {"kind": "dice", "p": 0.0}),
}


@torch.no_grad()
def run_adult_diag(cell, pol, wcfg, k, dev, pairs=PAIRS_ORIGINAL, n_per=4, T=1000, change_at=400, relearn=300, eval_seed=1000):
    """run_period の大人の時期と同じ始め方・同じ乱数の並びで回し、学び残しの材料を毎ステップ記録する"""
    k_before = Physio2.surprise_ahead
    Physio2.surprise_ahead = k
    B = n_per * len(pairs)
    P1 = torch.tensor([a for a, b in pairs for _ in range(n_per)], dtype=torch.float64)
    P2 = torch.tensor([b for a, b in pairs for _ in range(n_per)], dtype=torch.float64)
    g = torch.Generator(device=dev).manual_seed(eval_seed)
    g_pol = torch.Generator(device=dev).manual_seed(eval_seed + 7)
    ws = W.reset(wcfg, P1.to(dev), n=B, generator=g, device=dev)
    H = Wd = wcfg.size
    zone = W.zone_mask(wcfg, dev)
    wave = ~zone
    st = init_state(B, H, Wd, dev)
    frames = []
    for _ in range(N_HIST):
        frames.append(W.sense(ws, wcfg, g))
        ws = W.advance(ws)
    hist = torch.cat(frames[::-1], 1)
    wv = torch.tensor([exact_w(float(p)) for p in P1], device=dev, dtype=torch.float32).view(B, 3, 1, 1).expand(B, 3, H, Wd).clone()
    irr = torch.tensor([irreducible(wcfg.noise, float(p)) for p in P1], device=dev, dtype=torch.float32)
    v = (irr ** 2).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
    mu = wv[:, 0:1] * hist[:, 0:1] + wv[:, 1:2] * hist[:, 1:2] + wv[:, 2:3]
    if k == 1:
        vk = v.clone()
    else:
        gk = torch.tensor([1 + sum(x * x for x in noise_gain(float(p), k)) for p in P1], device=dev, dtype=torch.float32)
        vk = (wcfg.noise ** 2 * gk).clamp(min=Physio2.sigma_min ** 2).view(B, 1, 1, 1).expand(B, 1, H, Wd).clone()
    hs = [hist[:, i:i + 1] for i in range(N_HIST)]
    slots = [ahead(wv, hs[k - 1 - i], hs[k - i], k) if k - i <= N_HIST - 1 else ahead(wv, hs[N_HIST - 2], hs[N_HIST - 1], i + N_HIST - 1)
             for i in range(k)]
    st = st._replace(w=wv, v=v, hist=hist, mu=mu, vk=vk, buf=torch.cat(slots, 1))
    w1 = wv.clone()
    w2 = torch.tensor([exact_w(float(p)) for p in P2], device=dev, dtype=torch.float32).view(B, 3, 1, 1).expand(B, 3, H, Wd).clone()
    FI = torch.zeros(T, B, H, Wd, dtype=torch.bool, device=dev)
    NUM, DEN, Z1, ZK, NL = (torch.zeros(T, B, H, Wd, device=dev) for _ in range(5))
    for t in range(T):
        if t == change_at - 1:
            ws = W.set_period(ws, P2.to(dev))
        x1, x2 = st.hist[:, 0:1], st.hist[:, 1:2]                     # μ(t) を出した入力 [s(t−1), s(t−2)]
        m1 = w1[:, 0:1] * x1 + w1[:, 1:2] * x2 + w1[:, 2:3]
        m2 = w2[:, 0:1] * x1 + w2[:, 1:2] * x2 + w2[:, 2:3]
        NUM[t] = ((st.mu - m2) ** 2)[:, 0]
        DEN[t] = ((m1 - m2) ** 2)[:, 0]
        p = dict(pol)
        if p["kind"] == "dice":
            p["gen"] = g_pol
        if p["kind"] == "clock":
            p["t"] = t
        sense = W.sense(ws, wcfg, g)
        st, rec = cell.step(st, sense, policy=p)
        FI[t] = rec["hard"][:, 0]
        Z1[t] = rec["z2"][:, 0]
        ZK[t] = rec["zk2"][:, 0]
        NL[t] = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
        ws = W.advance(ws)
    Physio2.surprise_ahead = k_before

    names = [f"{a:g}→{b:g}" for a, b in pairs]
    per = {n: dict(stop=[], end=[], react=[]) for n in names}
    bins_z = np.zeros((len(BINS) - 1, 3))                             # 区間ごとの z₄² の和、z² の和、数
    q = []
    lag = wcfg.lag
    zero = torch.zeros(1, B, 1, device=dev)
    for x in range(Wd):
        rows = wave[:, x]
        if not bool(rows.any()):
            continue
        a0 = change_at + x * lag
        if a0 + relearn > T:
            continue
        col = lambda A_: A_[..., x][..., rows]                        # (T, B, 波の行)
        q.append(float(col(NL[a0:a0 + relearn]).mean()))
        Fw = col(FI[a0:a0 + relearn])
        tt = torch.arange(relearn, device=dev).view(relearn, 1, 1)
        last = torch.where(Fw, tt, torch.full_like(tt, -1).expand_as(Fw)).max(0).values          # (B, 行)。-1 は発火なし
        CN = torch.cat([zero.expand(1, B, int(rows.sum())), col(NUM).cumsum(0)], 0)             # (T+1, B, 行)
        CD = torch.cat([zero.expand(1, B, int(rows.sum())), col(DEN).cumsum(0)], 0)
        s0 = (a0 + last + 1).clamp(max=T)
        s1 = (s0 + 50).clamp(max=T)
        gat = lambda C, idx: C.gather(0, idx.unsqueeze(0)).squeeze(0)
        rem_stop = (gat(CN, s1) - gat(CN, s0)) / (gat(CD, s1) - gat(CD, s0)).clamp(min=1e-12)
        rem_end = (CN[a0 + relearn] - CN[a0 + relearn - 50]) / (CD[a0 + relearn] - CD[a0 + relearn - 50]).clamp(min=1e-12)
        react = last >= 0
        for i, n in enumerate(names):
            sl = slice(i * n_per, (i + 1) * n_per)
            per[n]["stop"] += rem_stop[sl][react[sl]].flatten().tolist()
            per[n]["end"] += rem_end[sl].flatten().tolist()
            per[n]["react"] += react[sl].flatten().float().tolist()
        # 補助：直近10ステップの学び残しで区間に分けた z₄²・z²（窓の中）
        tw = torch.arange(a0, a0 + relearn, device=dev)
        rN = CN[tw + 1] - CN[(tw - 9).clamp(min=0)]
        rD = (CD[tw + 1] - CD[(tw - 9).clamp(min=0)]).clamp(min=1e-12)
        r = (rN / rD).flatten().cpu().numpy()
        zk = col(ZK[a0:a0 + relearn]).flatten().cpu().numpy()
        z1 = col(Z1[a0:a0 + relearn]).flatten().cpu().numpy()
        idx = np.digitize(r, BINS[1:-1])
        for b_ in range(len(BINS) - 1):
            m = idx == b_
            bins_z[b_] += (zk[m].sum(), z1[m].sum(), m.sum())
    out = dict(fires=float(FI.sum()) / (B * H * Wd), quality=float(np.mean(q)), per_pair={})
    for n, d in per.items():
        out["per_pair"][n] = dict(stop_median=float(np.median(d["stop"])) if d["stop"] else None,
                                  end_median=float(np.median(d["end"])), react=float(np.mean(d["react"])), n_stop=len(d["stop"]))
    out["bins"] = [dict(lo=BINS[i], hi=BINS[i + 1], zk2=float(bins_z[i, 0] / max(bins_z[i, 2], 1)), z2=float(bins_z[i, 1] / max(bins_z[i, 2], 1)),
                        n=int(bins_z[i, 2])) for i in range(len(BINS) - 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(HERE / "runs" / "2b_B_seed0"), str(HERE / "runs" / "2b_B_seed1")])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = HERE / "runs" / "diagnose_b"
    out.mkdir(parents=True, exist_ok=True)
    res = dict(pairs=[list(p) for p in PAIRS_ORIGINAL], noises={})
    dnaless = AttentionCell(dna=False).to(dev).eval()
    sigma_before = Physio2.sigma_min
    for noise in (0.02, 0.005):
        Physio2.sigma_min = noise
        wcfg = W.WorldConfig(noise=noise)
        r = {}
        for name, (k, pol) in POLICIES.items():
            r[name] = run_adult_diag(dnaless, pol, wcfg, k, dev)
            print(f"ノイズ{noise}・{name}：済み", flush=True)
        # 一致の確かめ：B の発火数と NLL が run_period と同じ
        k_before = Physio2.surprise_ahead
        Physio2.surprise_ahead = 4                          # 4つの組の平均で比べる（quiet の組を分けない）（2026-09-30 の既定の切り替えのあとも、回したときと同じ値に固定。台帳 S2-39）
        rp_b = run_period(dnaless, {"kind": "trigger"}, wcfg, "adult", eval_seed=1000, device=dev, pairs=PAIRS_ORIGINAL, quiet=())
        Physio2.surprise_ahead = k_before
        r["_consistency_B"] = dict(diag=(r["B（4歩先、DNA なし）"]["fires"], r["B（4歩先、DNA なし）"]["quality"]),
                                   run_period=(rp_b["fires"], rp_b["quality"]),
                                   same=abs(rp_b["fires"] - r["B（4歩先、DNA なし）"]["fires"]) < 1e-6
                                   and abs(rp_b["quality"] - r["B（4歩先、DNA なし）"]["quality"]) < 1e-5)
        # 物差しの空の確かめ
        none = run_period(dnaless, {"kind": "dice", "p": 0.0}, wcfg, "adult", eval_seed=1000, device=dev, pairs=PAIRS_ORIGINAL, quiet=())
        r["_null"] = dict(gaveup=none["gaveup"], err_back=none["err_back"], quality=none["quality"], per_pair=none["per_pair"])
        if noise == 0.02:
            for run in map(Path, args.runs):
                if not (run / "model.pt").exists():
                    continue
                m = AttentionCell(dna=True).to(dev).eval()
                m.load_state_dict(torch.load(run / "model.pt", map_location=dev))
                cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
                r[f"B（学んだ DNA、シード{cfg['seed']}）"] = run_adult_diag(m, {"kind": "trigger"}, wcfg, 4, dev)
                print(f"ノイズ{noise}・学んだ DNA シード{cfg['seed']}：済み", flush=True)
        res["noises"][str(noise)] = r
    Physio2.sigma_min = sigma_before

    # 判定（15章）
    def floor(r, n):
        return r["密（DNA なし）"]["per_pair"][n]["end_median"]
    r5 = res["noises"]["0.005"]
    fl = {n: floor(r5, n) for n in JUDGE}
    b5 = {n: r5["B（4歩先、DNA なし）"]["per_pair"][n]["stop_median"] for n in JUDGE}
    o5 = {n: r5["1歩先の驚き（DNA なし）"]["per_pair"][n]["stop_median"] for n in JUDGE}
    support = all(b5[n] is not None and b5[n] <= fl[n] + 0.1 and o5[n] is not None and o5[n] >= fl[n] + 0.3 for n in JUDGE)
    against = any(b5[n] is not None and b5[n] >= fl[n] + 0.3 for n in JUDGE)
    verdict = "仮説を支持" if support else ("仮説に反する" if against else "どちらとも言えない")
    null_gave = res["noises"]["0.02"]["_null"]["gaveup"]
    res["verdict"] = dict(hypothesis=verdict, floor_0005=fl, B_stop_0005=b5, one_step_stop_0005=o5,
                          metric_lenient_002=null_gave <= 0.05, null_gaveup_002=null_gave)
    r2 = res["noises"]["0.02"]
    fl2 = {n: floor(r2, n) for n in JUDGE}
    res["verdict"]["noise_002"] = {who: {n: (r2[who]["per_pair"][n]["stop_median"], fl2[n]) for n in JUDGE}
                                   for who in r2 if who.startswith("B")}
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 仮説「遠くを予測した驚きは、学び終わりの合図にもなる」の確かめ（DESIGN.md 15章）", "",
         f"**判定（決め手はノイズ0.005）：{verdict}**", "",
         f"物差しの空の確かめ：ノイズ0.02 で一度も書き換えないときのあきらめ {null_gave:.2f}"
         + (" → **ノイズ0.02 のあきらめ0 は、それだけでは学び終えた証拠にならない（物差しが甘い）**" if null_gave <= 0.05 else ""), ""]
    for noise, r in res["noises"].items():
        L += [f"## ノイズ{noise}", "", f"一致の確かめ（B の発火数・NLL が run_period と同じ）：{'一致' if r['_consistency_B']['same'] else '不一致'}", "",
              "| 回し方 | 組 | 反応したセル | 止まったときの学び残し（中央値） | 窓の終わりの学び残し（中央値） |", "|---|---|---:|---:|---:|"]
        for who, d in r.items():
            if who.startswith("_"):
                continue
            for n, pp in d["per_pair"].items():
                s = "—" if pp["stop_median"] is None else f"{pp['stop_median']:.3f}"
                L.append(f"| {who} | {n} | {pp['react']:.2f} | {s} | {pp['end_median']:.3f} |")
        nl = r["_null"]
        L += ["", f"書き換えない：あきらめ {nl['gaveup']:.2f}、戻り {nl['err_back']:.2f}、学び直しの NLL {nl['quality']:.3f}", "",
              "補助（B の回し方で、直近10ステップの学び残しの区間ごとの z₄² と z² の平均）：", "", "| 学び残し | z₄² | z² | ステップ×セル |", "|---|---:|---:|---:|"]
        for b_ in r["B（4歩先、DNA なし）"]["bins"]:
            L.append(f"| {b_['lo']:g}〜{b_['hi']:g} | {b_['zk2']:.2f} | {b_['z2']:.2f} | {b_['n']} |")
        L.append("")
    L += ["床＝密（毎ステップ書き換える）の窓の終わりの学び残し。判定のしきい値は床からの差（床＋0.1 以下＝学び終えてから止まる、床＋0.3 以上＝学び終える前に止まる）", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
