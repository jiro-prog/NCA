"""
睡眠の模擬の追加の診断（step3_memory/DESIGN.md 9.6。読み方は走らせる前に書いた）。

    .venv/Scripts/python.exe step3_memory/precheck/sleep_followup.py [--device cuda]

(a) ③（γ_L=0.99）の寝る規則だけを「静かな間は、最後の驚きの発火から100ステップごとに見直す」に変えた版（③周期）で、
    見逃したセルが後の切り替えでまた反応するか（absorb_check.py と同じ数え方）と、9.3 の判定の数字
(b) ③（寝る規則はそのまま）で、見逃しが 12/20 の組に多い理由：切り替えの直前の σ₄ 比（セルごと）の分布と、区間ごとの見逃しの割合（判定なし）
出力：step3_memory/runs/precheck_sleep/followup.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import headroom as H  # noqa: E402
import sleep as S  # noqa: E402
from headroom import W  # noqa: E402

BINS = (0.0, 1.1, 1.2, 1.3, 1.5, 2.0, float("inf"))


def absorb(res, start, gi):
    R = res["react"][:, :, res["wave"]]
    first = 1 if start == "adult" else 0
    r = R[first:, gi * H.N_PER:(gi + 1) * H.N_PER]
    missed = ~r
    n_sw = r.shape[0]
    idx = torch.arange(n_sw, device=r.device).view(-1, 1, 1)
    first_miss = torch.where(missed, idx, torch.full_like(idx, n_sw).expand_as(missed)).min(0).values
    ever = missed.any(0)
    back = (r & (idx > first_miss.unsqueeze(0))).any(0) & ever
    return int(ever.sum()), int(back.sum()), r.shape[1] * r.shape[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    v3 = [x for x in S.variants() if x["name"] == "③(γ=0.99)"][0]
    v3p = dict(v3, name="③周期(γ=0.99)", periodic=True)
    L = ["# 睡眠の模擬の追加の診断（DESIGN.md 9.6。読み方は走らせる前に書いた）", "",
         "## (a) 寝る規則を「静かな間は100ステップごと」にすると、見逃したセルは戻るか", "",
         "| 始め方 | 評価の乱数 | 組 | ③：見逃した／後で反応 | ③周期：見逃した／後で反応 |", "|---|---:|---|---|---|"]
    allres = {}
    rates = []
    per_cell = {}
    for start in S.STARTS:
        for seed in H.EVAL_SEEDS:
            r3 = S.run(v3, wcfg, seed, args.device, start)
            r3p = S.run(v3p, wcfg, seed, args.device, start)
            per_cell[(start, seed)] = r3
            allres[f"{start}|{seed}|③周期(γ=0.99)"] = dict(groups=S.summarize(r3p, wcfg))
            for gi, grp in enumerate(H.GROUPS):
                e3, b3, n = absorb(r3, start, gi)
                ep, bp, _ = absorb(r3p, start, gi)
                rates.append(bp / max(ep, 1))
                L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {'/'.join(f'{p:g}' for p in grp)} | "
                         f"{e3 / n:.3f}／{b3 / max(e3, 1):.2f} | {ep / n:.3f}／{bp / max(ep, 1):.2f} |")
    j = S.judge(allres, "③周期(γ=0.99)", H.EVAL_SEEDS)
    mn = min(rates)
    verdict = ("戻らない原因は寝る規則（静かな時期に1回だけ）" if mn >= 0.8 else
               ("寝る規則だけでは説明できない（別の原因がある）" if mn < 0.5 else "一部は寝る規則"))
    L += ["", f"- 見逃したセルのうち後で反応した割合（③周期）の最小：{mn:.2f} → **{verdict}**（9.6 の読み方）",
          f"- ③周期の 9.3 の数字：判定 {j['verdict']}{'・落ち着かない' if j['restless_flag'] else ''}、反応（最小／平均）{j['react_min']:.2f}／{j['react_mean']:.2f}、"
          f"落ち着かなさ {j['restless']:.3f}、驚きの発火／切り替え {j['fires']:.2f}、睡眠の書き込み／切り替え {j['sleeps']:.2f}、直前の σ₄ 比 {j['sig_ratio']:.2f}、窓の NLL {j['quality']:.3f}",
          "", "## (b) ③ で、見逃しが 12/20 の組に多い理由（判定なし）", "",
          "判定する切り替えをまとめて、切り替えの直前の σ₄ 比（セルごと。古い周期の正確な係数での定常の大きさとの比）の区間ごとに、見逃した割合（セルの数）", "",
          "| 組 | 切り替え | σ₄ 比の中央値／90% 点 | " + " | ".join(f"{BINS[i]:g}〜{BINS[i + 1]:g}" for i in range(len(BINS) - 1)) + " |",
          "|---|---|---|" + "---:|" * (len(BINS) - 1)]
    for gi, grp in enumerate(H.GROUPS):
        for kind in ("B→A", "A→B", "その他"):
            ratios, misses = [], []
            for (start, seed), res in per_cell.items():
                sl = slice(gi * H.N_PER, (gi + 1) * H.N_PER)
                first = 1 if start == "adult" else 0
                for sw in range(first, len(H.SEQ) - 1):
                    key = f"{H.SEQ[sw]}→{H.SEQ[sw + 1]}"
                    k = key if key in ("B→A", "A→B") else "その他"
                    if k != kind:
                        continue
                    P_old = float(res["sched"][sw][sl][0])
                    ratios.append((res["sig_pre"][sw][sl][:, res["wave"]] / S.sig4_exact(wcfg.noise, P_old)).flatten().cpu().numpy())
                    misses.append((~res["react"][sw][sl][:, res["wave"]]).flatten().cpu().numpy())
            ra = np.concatenate(ratios)
            mi = np.concatenate(misses)
            idx = np.digitize(ra, BINS[1:-1])
            cells = []
            for b_ in range(len(BINS) - 1):
                m = idx == b_
                cells.append(f"{mi[m].mean():.2f}（{m.sum()}）" if m.sum() else "—")
            L.append(f"| {'/'.join(f'{p:g}' for p in grp)} | {kind} | {np.median(ra):.2f}／{np.quantile(ra, 0.9):.2f} | " + " | ".join(cells) + " |")
    path = H.ROOT / "step3_memory" / "runs" / "precheck_sleep" / "followup.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
