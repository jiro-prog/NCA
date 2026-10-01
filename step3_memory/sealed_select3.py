"""
Step 3 の封印の周期を選ぶ（DESIGN.md 10.3 の決まり。式だけで、性能は見ない）。

    .venv/Scripts/python.exe step3_memory/sealed_select3.py

決まり（10.3、2026-10-01・02 に決めた）：
  - 物差しは Step 2 の封印の周期の選び方と同じ**静的な z₄²**（学び終えた係数のまま周期を変えた直後の4歩先の驚きの平均。step2_attention/sealed_select.z4_static）。ノイズ0.02
  - 開発の2つの組（12/24/8 と 12/20/8）それぞれと同じくらいの見えにくさの組を、1つずつ選ぶ：並び A, B, A, B, A, C, A, B に出てくる切り替え4種類
    （A→B、B→A、A→C、C→A）の静的な z₄² の対数の差の2乗の和がいちばん小さい組
  - C の条件：C の |cos ω の差| が、A・B のどちらからも 0.159 以上（開発の C=8 と A=12 の差）
  - 除く周期：Step 3 の開発の周期 {8, 12, 20, 24}、Step 2 の評価と封印の周期 {11, 14, 19, 26, 36, 40}。候補は整数の6〜48。A・B・C はたがいに違う
  - 同じなら周期の小さいほう（A、B、C の順に比べる）。2つの組は違う組にする（1つ目で選んだ組は、2つ目では選ばない）
出力：step3_memory/runs/sealed_select3/（summary.md、summary.json）。結果は sealed3.py に書き、以後触らない
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step2_attention"))

from sealed_select import z4_static  # noqa: E402

NOISE = 0.02
DEV_GROUPS = ((12, 24, 8), (12, 20, 8))
EXCLUDE = {8, 12, 20, 24} | {11, 14, 19, 26, 36, 40}
CANDS = [p for p in range(6, 49) if p not in EXCLUDE]
C_MIN = 0.159
SWITCHES = (("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"))


def profile(g):
    P = dict(zip("ABC", g))
    return [math.log(z4_static(P[a], P[b], NOISE)) for a, b in SWITCHES]


def dcos(p, q):
    return abs(math.cos(2 * math.pi / p) - math.cos(2 * math.pi / q))


def main():
    chosen, rows = [], []
    for dev in DEV_GROUPS:
        target = profile(dev)
        best = None
        for A in CANDS:
            for B in CANDS:
                if B == A:
                    continue
                for C in CANDS:
                    if C in (A, B) or min(dcos(C, A), dcos(C, B)) < C_MIN:
                        continue
                    if (A, B, C) in chosen:
                        continue
                    pr = profile((A, B, C))
                    score = sum((x - y) ** 2 for x, y in zip(pr, target))
                    key = (score, A, B, C)
                    if best is None or key < best[0]:
                        best = (key, (A, B, C), pr)
        (score, A, B, C), grp, pr = best[0], best[1], best[2]
        chosen.append(grp)
        rows.append(dict(dev=dev, sealed=grp, score=score,
                         dev_z4={f"{a}→{b}": round(math.exp(v), 1) for (a, b), v in zip(SWITCHES, target)},
                         sealed_z4={f"{a}→{b}": round(math.exp(v), 1) for (a, b), v in zip(SWITCHES, pr)},
                         c_dcos=round(min(dcos(grp[2], grp[0]), dcos(grp[2], grp[1])), 3)))
    out = HERE / "runs" / "sealed_select3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(rule=dict(noise=NOISE, exclude=sorted(EXCLUDE), cands=CANDS, c_min=C_MIN), chosen=rows),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Step 3 の封印の周期（DESIGN.md 10.3 の決まり。式だけで、性能は見ない）", "",
         "| 開発の組 | 選んだ封印の組 | 静的な z₄²（A→B／B→A／A→C／C→A）開発 | 同じく封印 | C と A・B の |cos ω の差| の小さいほう |", "|---|---|---|---|---:|"]
    for r in rows:
        d, sz = r["dev_z4"], r["sealed_z4"]
        L.append(f"| {'/'.join(map(str, r['dev']))} | **{'/'.join(map(str, r['sealed']))}** | {'／'.join(str(v) for v in d.values())} | {'／'.join(str(v) for v in sz.values())} | {r['c_dcos']} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
