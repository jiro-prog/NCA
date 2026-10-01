"""
回帰の確かめ：睡眠の段を本物のセル（cell.Cell3）に入れたものが、模擬（precheck/sleep.py）と同じ数字になるか（2026-10-01、そうの判断：会計のテストの前に）。

    .venv/Scripts/python.exe step3_memory/check_cell.py [--device cuda]

同じ世界・同じ乱数（評価の乱数300・301、始め方は大人と育ち）で、
  床   ：模擬の「床」          と Cell3(sleep=False)
  天井 ：模擬の「天井」        と Cell3(sleep=False)＋evaluate.run(ceiling=True)
  眠る段：模擬の「(は) 2倍ずつ」（③・γ_L=0.99・間隔を2倍ずつ）と Cell3(sleep=True)
  記憶の段：思い出し方の模擬（precheck/recall.py）の「(お1)」と Cell3(sleep=True, memory=True)（2026-10-02 に足した。保存先・数え・最後に使ってからの状態と、
          見張りの記録（思い出し・取り違えの内訳・何回目の発火か・比べた保存先の数・いっぱい・入れ替え）も比べる）
を回し、セルごとの記録（反応・驚きの発火・睡眠の書き込み・窓の NLL・学び残しのもと・σ₄ など）、まとめの数字、最後の状態のすべての差を調べる。
合格：すべての差が0（1ビットも同じ）。出力：step3_memory/runs/check_cell/summary.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "precheck"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import cell as C  # noqa: E402
import evaluate as E  # noqa: E402
import recall as RC  # noqa: E402
import sleep as S  # noqa: E402
from headroom import W  # noqa: E402

KEYS = ("react", "fires_ph", "sleeps_ph", "out_fires", "nll_sum", "num_end", "den_end", "sig_pre", "fires_tot", "sleeps_tot")
MEM_KEYS = ("react", "fires_ph", "sleeps_ph", "recalls_ph", "nll_sum", "mis_novel", "mis_return", "mis_noop", "mis_change", "rec_ok", "ever_full", "evict")


def diff(a, b):
    if a.dtype == torch.bool:
        return float((a != b).sum())
    return float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    wcfg = W.WorldConfig(noise=0.02)
    pairs = [
        ("床", dict(name="床", kind="floor"), dict(sleep=False, ceiling=False)),
        ("天井", dict(name="天井", kind="ceiling"), dict(sleep=False, ceiling=True)),
        ("眠る段", dict(name="(は) 2倍ずつ", kind="both", g=0.99, doubling=True), dict(sleep=True, ceiling=False)),
    ]
    L = ["# 回帰の確かめ：本物のセル（cell.Cell3）と模擬（precheck/sleep.py）", "",
         "差は、セルごとの記録の差の最大（真偽の記録は、食い違ったセルの数）、まとめの数字の差の最大、最後の状態のすべて（係数・痕跡・σ・σ₄・濃度・不応期・最近の値・予測・覚えている予測・長い痕跡・数え・見直しの回数）の差の最大。合格はすべて0", "",
         "| 始め方 | 評価の乱数 | 段 | " + " | ".join(KEYS) + " | まとめの数字 | 最後の状態のすべて |", "|---|---:|---|" + "---:|" * (len(KEYS) + 2)]
    worst = 0.0
    for start in ("adult", "grow"):
        for seed in (300, 301):
            for name, var, opt in pairs:
                ref = S.run(var, wcfg, seed, dev, start)
                new = E.run(C.Cell3(sleep=opt["sleep"], device=dev), wcfg, seed, start, dev=dev, ceiling=opt["ceiling"])
                ds = [diff(ref[k], new[k]) for k in KEYS]
                fr, fn = ref["final"], new["final"]                  # 最後の状態のすべて（床の状態の各欄＋長い痕跡・数え・見直しの回数）
                dst = max([diff(a_, b_) for a_, b_ in zip(fr["core"], fn.core)] +
                          [diff(fr["RL"], fn.RL), diff(fr["rL"], fn.rL), diff(fr["q"], fn.q), diff(fr["k"], fn.k)])
                sr, sn = S.summarize(ref, wcfg), E.summarize(new, wcfg)
                dsum = 0.0
                for gr, gn in zip(sr, sn):
                    dsum = max(dsum, abs(gr["fires_life"] - gn["fires_life"]), abs(gr["sleeps_life"] - gn["sleeps_life"]))
                    for rr, rn in zip(gr["rows"], gn["rows"]):
                        for key in ("react", "sig_ratio", "fires", "sleeps", "restless", "quality", "resid"):
                            dsum = max(dsum, abs(rr[key] - rn[key]))
                worst = max(worst, max(ds), dsum, dst)
                L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {name} | " + " | ".join(f"{d:g}" for d in ds) + f" | {dsum:g} | {dst:g} |")
                print(f"{start} {seed} {name}: 最大の差 {max(max(ds), dsum, dst):g}", flush=True)
    # 記憶の段（模擬の (お1) と本物のセル）
    L += ["", "## 記憶の段：思い出し方の模擬の (お1) と Cell3(sleep=True, memory=True)", "",
          "| 始め方 | 評価の乱数 | " + " | ".join(MEM_KEYS) + " | 何回目の発火か | 比べた保存先の数 | 最後の状態のすべて（保存先を含む） |",
          "|---|---:|" + "---:|" * (len(MEM_KEYS) + 3)]
    for start in ("adult", "grow"):
        for seed in (300, 301):
            ref = RC.run("お1", wcfg, seed, start, dev)
            new = E.run(C.Cell3(sleep=True, memory=True, device=dev), wcfg, seed, start, dev=dev)
            ds = [diff(ref[k], new[k]) for k in MEM_KEYS]
            dnth = max(abs(a_ - b_) for a_, b_ in zip(ref["nth_hist"], new["nth_hist"]))
            dcomp = abs(ref["comp"] - new["comp"])
            fr, fn = ref["final"], new["final"]
            dst = max([diff(a_, b_) for a_, b_ in zip(fr["core"], fn.core)] +
                      [diff(fr["RL"], fn.RL), diff(fr["rL"], fn.rL), diff(fr["q"], fn.q), diff(fr["k"], fn.k),
                       diff(fr["slot"], fn.mslot), diff(fr["n"], fn.mn), diff(fr["age"], fn.mage)])
            worst = max(worst, max(ds), dnth, dcomp, dst)
            L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | " + " | ".join(f"{d:g}" for d in ds) + f" | {dnth:g} | {dcomp:g} | {dst:g} |")
            print(f"{start} {seed} 記憶の段: 最大の差 {max(max(ds), dnth, dcomp, dst):g}", flush=True)
    verdict = "合格（1ビットも同じ）" if worst == 0.0 else f"不合格（最大の差 {worst:g}）"
    L += ["", f"**{verdict}**"]
    out = HERE / "runs" / "check_cell"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if worst != 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
