"""
封印の周期の選び直し（DESIGN.md 7.4 の「2026-09-30 の直し」。決まりはこの計算より先に書いてコミットした。台帳 S2-42）。

    .venv/Scripts/python.exe step2_attention/sealed_select.py

性能は見ない。計算するのは、学び終えた予測器（P₁ の正確な係数）を止めたまま、周期を P₂ に変えた直後の4歩先の z₄² の平均（位相で平均）だけ：
    z₄² ＝ 1 ＋ (A²/2)·|e^{4iω₂} − α₄ − β₄e^{−iω₂}|² ／ (σ_n²(1 ＋ α₄² ＋ β₄²))
A=0.4（波の振幅）、ω₂ = 2π/P₂、(α₄, β₄) は P₁ の正確な係数を4回まわしたときの入力の係数（protocol.noise_gain）。分母は変える前の σ₄² の定常の大きさ。
決まり：役目は3つ（a. 行き先が内側の穴、b. 学習の範囲の外の短い周期、c. 学習の範囲の外の長い周期）。候補は整数、P₁ は学習の範囲（10〜17、23〜30）、
開発の評価の周期 12・20・24・36 は使わない。見える＝ノイズ0.025 で z₄² ≥ 10。同じ役目の候補がいくつもあれば、前の封印の変化にいちばん近いもの
（(P₁, P₂) の距離。同じなら P₁、次に P₂ の小さいほう）。
出力：runs/sealed_select/（summary.json、summary.md）。この結果を sealed.py に書いたら、以後は封印の定義を変えない
"""
import cmath
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import noise_gain  # noqa: E402

A = 0.4
K = 4
NOISE_SELECT = 0.025                       # 封印の世界でいちばんノイズの大きい S3
Z_MIN = 10.0
DEV_PERIODS = {12, 20, 24, 36}
P1_RANGE = [p for p in list(range(10, 18)) + list(range(23, 31)) if p not in DEV_PERIODS]
ROLES = {
    "a": dict(desc="行き先が内側の穴（18〜22 の中）", P2=[19, 21], old=(11, 19)),
    "b": dict(desc="行き先が学習の範囲の外（短い周期）", P2=[5, 6, 7, 8, 9], old=(26, 8)),
    "c": dict(desc="行き先が学習の範囲の外（長い周期）", P2=[p for p in range(31, 61) if p not in DEV_PERIODS], old=(14, 40)),
}


def z4_static(P1, P2, noise):
    """学び終えた予測器（P₁）を止めたまま、周期を P₂ に変えた直後の4歩先の z₄² の平均（位相で平均）"""
    a4, b4 = noise_gain(float(P1), K)
    w2 = 2 * math.pi / P2
    ph = cmath.exp(1j * K * w2) - a4 - b4 * cmath.exp(-1j * w2)
    clean = A ** 2 / 2 * abs(ph) ** 2
    return 1 + clean / (noise ** 2 * (1 + a4 ** 2 + b4 ** 2))


def main():
    # 式の確かめ（DESIGN.md 7.4 に書いた、REVIEW.md §5 の見積もりとの突き合わせ）
    check = dict(z4_12to24_noise002=z4_static(12, 24, 0.02))
    res = dict(rule=dict(k=K, noise=NOISE_SELECT, z_min=Z_MIN, p1_range=P1_RANGE, dev_periods=sorted(DEV_PERIODS),
                         roles={r: dict(desc=d["desc"], P2=d["P2"], old=d["old"]) for r, d in ROLES.items()}),
               formula_check=check, chosen={}, candidates={})
    for role, d in ROLES.items():
        cands = []
        for P1 in P1_RANGE:
            for P2 in d["P2"]:
                z = z4_static(P1, P2, NOISE_SELECT)
                cands.append(dict(P1=P1, P2=P2, z4_noise0025=z, z4_noise002=z4_static(P1, P2, 0.02), visible=z >= Z_MIN,
                                  dist=math.hypot(P1 - d["old"][0], P2 - d["old"][1])))
        ok = [c for c in cands if c["visible"]]
        best = min(ok, key=lambda c: (c["dist"], c["P1"], c["P2"])) if ok else None
        res["candidates"][role] = cands
        res["chosen"][role] = best
    out = HERE / "runs" / "sealed_select"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# 封印の周期の選び直し（DESIGN.md 7.4 の決まり。性能は見ない、式だけ）", "",
         f"式の確かめ：12→24（ノイズ0.02）の静的な z₄² は {check['z4_12to24_noise002']:.1f}（REVIEW.md §5 の見積もりは64.5）", "",
         "| 役目 | 前の変化 | 選んだ変化 | z₄²（ノイズ0.025） | z₄²（ノイズ0.02） | 見える候補の数 |", "|---|---|---|---:|---:|---:|"]
    for role, d in ROLES.items():
        c = res["chosen"][role]
        n_ok = sum(x["visible"] for x in res["candidates"][role])
        old = res["candidates"][role]
        old_c = next(x for x in old if (x["P1"], x["P2"]) == d["old"]) if d["old"][0] in P1_RANGE else None
        old_s = f"{d['old'][0]}→{d['old'][1]}" + (f"（z₄² {old_c['z4_noise0025']:.1f}）" if old_c else "")
        L.append(f"| {role}. {d['desc']} | {old_s} | **{c['P1']}→{c['P2']}** | {c['z4_noise0025']:.1f} | {c['z4_noise002']:.1f} | {n_ok} |" if c else
                 f"| {role}. {d['desc']} | {old_s} | （見える候補なし） | — | — | 0 |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
