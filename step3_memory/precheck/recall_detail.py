"""
思い出し方の模擬の補い（読み方は事前に決めていない。判定には使わない）：(お1) の取り違えを、害のないものと害のあるものに分ける。

    .venv/Scripts/python.exe step3_memory/precheck/recall_detail.py [--device cuda]

取り違え（思い出した保存先が、その列がいま見ているパターンと違う）を、思い出した保存先 u と、驚いた瞬間の書き換える前の係数 w_今 の、最近の材料の上での差
(u − w_今)ᵀR(u − w_今) が 6σ² 以下（見分けがつかない＝書き直しても予測はほぼ変わらない）か、超える（違う係数を書く＝予測が変わる）かで分ける。
recall.py と同じ回し方・同じ乱数（セルの動きは recall.py の (お1) と同じ）。出力：step3_memory/runs/precheck_recall/detail.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import recall as RC  # noqa: E402
from storage_check import W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    L = ["# (お1) の取り違えの中身（思い出し方の模擬の補い。読み方は事前に決めていない）", "",
         "見分けがつかない＝思い出した保存先といまの係数の、最近の材料の上での差が6σ² 以下（書き直しても予測はほぼ変わらない）。違う係数＝それを超える（予測が変わる）", "",
         "| 始め方 | 評価の乱数 | 取り違え | うち見分けがつかない（書き直し） | うち違う係数を書いた | 正しく思い出した |", "|---|---:|---:|---:|---:|---:|"]
    tot = [0, 0, 0, 0]
    for start in RC.STARTS:
        for seed in RC.SEEDS:
            r = RC.run("お1", wcfg, seed, start, args.device)
            wv = r["wave"]
            mis = int(r["mis_novel"][:, wv].sum() + r["mis_return"][:, wv].sum())
            noop = int(r["mis_noop"][:, wv].sum())
            chg = int(r["mis_change"][:, wv].sum())
            ok = int(r["rec_ok"][:, wv].sum())
            tot = [a + b for a, b in zip(tot, (mis, noop, chg, ok))]
            L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {mis} | {noop} | {chg} | {ok} |")
            print(start, seed, "done", flush=True)
    L.append(f"| 合計 | | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} |")
    out = HERE.parent / "runs" / "precheck_recall" / "detail.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
