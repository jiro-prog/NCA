"""
睡眠の模擬の気づきの確かめ：一度切り替えを見逃したセルは、見逃したままになるか（読み方は事前に決めていない。判定には使わない）。

    .venv/Scripts/python.exe step3_memory/precheck/absorb_check.py [--device cuda]

sleep.py の ③・②（γ_L=0.99）を同じ世界・同じ乱数で回し、判定する切り替え（大人は2回目以降、育ちはすべて）のどれかを見逃した（窓の中で一度も驚いて発火しなかった）
波の区域のセルの割合と、そのうち後の切り替えでまた反応したセルの数を数える。出力：step3_memory/runs/precheck_sleep/absorb_check.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import headroom as H  # noqa: E402
import sleep as S  # noqa: E402
from headroom import W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    L = ["# 一度見逃したセルは、見逃したままになるか（睡眠の模擬の気づきの確かめ。読み方は事前に決めていない）", "",
         "| 始め方 | 版 | 評価の乱数 | 組 | 見逃したことのあるセル | そのうち後でまた反応したセル |", "|---|---|---:|---|---:|---:|"]
    for start in S.STARTS:
        for vname in ("③(γ=0.99)", "②(γ=0.99)"):
            v = [x for x in S.variants() if x["name"] == vname][0]
            for seed in H.EVAL_SEEDS:
                res = S.run(v, wcfg, seed, args.device, start)
                R = res["react"][:, :, res["wave"]]                                     # (切り替え, 世界, 波の区域のセル)
                first = 1 if start == "adult" else 0
                for gi, grp in enumerate(H.GROUPS):
                    r = R[first:, gi * H.N_PER:(gi + 1) * H.N_PER]
                    missed = ~r
                    n_sw = r.shape[0]
                    idx = torch.arange(n_sw, device=r.device).view(-1, 1, 1)
                    first_miss = torch.where(missed, idx, torch.full_like(idx, n_sw).expand_as(missed)).min(0).values
                    ever = missed.any(0)
                    back = (r & (idx > first_miss.unsqueeze(0))).any(0) & ever
                    n = r.shape[1] * r.shape[2]
                    L.append(f"| {'大人' if start == 'adult' else '育ち'} | {vname} | {seed} | {'/'.join(f'{p:g}' for p in grp)} | "
                             f"{int(ever.sum())}/{n}（{float(ever.sum()) / n:.3f}） | {int(back.sum())}（{float(back.sum()) / max(int(ever.sum()), 1):.2f}） |")
    path = H.ROOT / "step3_memory" / "runs" / "precheck_sleep" / "absorb_check.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
