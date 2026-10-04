"""
Step 3b の封印の周期を選ぶ（step3b_quiet/DESIGN.md 4章の決まり。式だけで、性能は見ない。2026-10-04、そうの確認のあとで一度だけ計算する）。

    .venv/Scripts/python.exe step3b_quiet/sealed_select3b.py

決まり（4章）：
  - 候補は整数の6〜59。除く周期：Step 3 の封印 {7, 13, 30, 48}、Step 3 の確かめで見た38、Step 3 の開発 {8, 12, 20, 24}、Step 2 の評価と封印 {11, 14, 19, 26, 36, 40}、
    Step 3b の開発 {8, 12, 24, 30, 34, 35, 38, 48, 60}
  - v₄（ノイズ0.02）の帯：見えやすい・遠い EF（15 以上）、見えやすい・近い EN（10.5〜13）、見えにくい・近い HN（8.5〜9.8）、見えにくい・遠い HF（7 以下）
  - 組1：A∈EF、B∈HF、C∈EN。組2：A∈EF（組1 と違う）、B∈HN、C∈EN（組1 と違う）。6つの周期はすべて違う
  - 並びに出てくる切り替え4種類（A→B、B→A、A→C、C→A）の静的な z₄²（step2_attention/sealed_select.z4_static）が、ノイズ0.025 で10 以上
  - 同じ条件を満たすものが複数あれば、帯の真ん中（EF 20、EN 11.75、HN 9.15、HF 6.3）からの v₄ の差（ノイズ0.02）の合計がいちばん小さい組、それでも同じなら (A, B, C) の小さいほう。
    組1 を先に選び、組2 は残りから選ぶ
出力：step3b_quiet/runs/sealed_select3b/（summary.md、summary.json）。結果は sealed3b.py に書き、以後触らない
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step2_attention"))

from sealed_select import z4_static  # noqa: E402

AMP = 0.4
EXCLUDE = {7, 13, 30, 48} | {38} | {8, 12, 20, 24} | {11, 14, 19, 26, 36, 40} | {8, 12, 24, 30, 34, 35, 38, 48, 60}
CANDS = [p for p in range(6, 60) if p not in EXCLUDE]
BANDS = {"EF": (15.0, 1e9, 20.0), "EN": (10.5, 13.0, 11.75), "HN": (8.5, 9.8, 9.15), "HF": (0.0, 7.0, 6.3)}
GROUP_BANDS = (("EF", "HF", "EN"), ("EF", "HN", "EN"))
VIS_NOISE, VIS_MIN = 0.025, 10.0
SWITCHES = (("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"))


def v4(P, noise):
    return math.sqrt(2) * AMP * abs(math.sin(4 * math.pi / P)) / noise


def in_band(P, band):
    lo, hi, _ = BANDS[band]
    return lo <= v4(P, 0.02) <= hi


def pick(bands, used):
    best = None
    pa = [p for p in CANDS if in_band(p, bands[0]) and p not in used]
    pb = [p for p in CANDS if in_band(p, bands[1]) and p not in used]
    pc = [p for p in CANDS if in_band(p, bands[2]) and p not in used]
    for A in pa:
        for B in pb:
            for C in pc:
                if len({A, B, C}) < 3:
                    continue
                P = {"A": A, "B": B, "C": C}
                z = [z4_static(P[a], P[b], VIS_NOISE) for a, b in SWITCHES]
                if min(z) < VIS_MIN:
                    continue
                score = sum(abs(v4(p, 0.02) - BANDS[bd][2]) for p, bd in zip((A, B, C), bands))
                key = (round(score, 9), A, B, C)
                if best is None or key < best[0]:
                    best = (key, (A, B, C), z)
    return best


def main():
    chosen, rows, used = [], [], set()
    for bands in GROUP_BANDS:
        best = pick(bands, used)
        if best is None:
            raise SystemExit(f"帯 {bands} を満たす組がない")
        (score, A, B, C), grp, z = best
        chosen.append(grp)
        used |= set(grp)
        rows.append(dict(bands=list(bands), group=list(grp), score=score,
                         v4_002={p: round(v4(p, 0.02), 2) for p in grp}, v4_0025={p: round(v4(p, 0.025), 2) for p in grp},
                         z4_0025={f"{a}→{b}": round(x, 1) for (a, b), x in zip(SWITCHES, z)}))
    out = HERE / "runs" / "sealed_select3b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(rule=dict(exclude=sorted(EXCLUDE), cands=CANDS, bands=BANDS, vis=(VIS_NOISE, VIS_MIN)), chosen=rows),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Step 3b の封印の周期（DESIGN.md 4章の決まり。式だけで、性能は見ない）", "",
         "| 組 | 帯（A/B/C） | 選んだ周期（A/B/C） | v₄（ノイズ0.02） | v₄（ノイズ0.025） | 静的な z₄²（ノイズ0.025。A→B／B→A／A→C／C→A） |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        g = r["group"]
        L.append(f"| {i} | {'/'.join(r['bands'])} | **{'/'.join(map(str, g))}** | {'／'.join(str(r['v4_002'][p]) for p in g)} | "
                 f"{'／'.join(str(r['v4_0025'][p]) for p in g)} | {'／'.join(str(x) for x in r['z4_0025'].values())} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
