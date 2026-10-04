"""
取り違えの物差しの確かめ (a)：ノイズの乗った入力から学ぶ最小二乗の係数のずれを、式で予想する（step3_memory/DESIGN.md 16章。2026-10-04、読み方は走らせる前に書いた）。回さない。

    .venv/Scripts/python.exe step3_memory/precheck/ls_bias.py

予測器 μ(t+1) = w₁s(t) + w₂s(t−1) + b を、s = 0.5 + 0.4 sin(ωt) + N(0, σ_n²) から十分長い材料の最小二乗で解いた係数 w_LS(P, σ_n)：
  [γ₀ γ₁; γ₁ γ₀][w₁; w₂] = [γ₁; γ₂]、γ₀ = A²/2 + σ_n²、γ_k = (A²/2)cos(kω)、b = 0.5(1 − w₁ − w₂)
g_a ＝ w_LS と正確な係数の、きれいな波の上での1歩先の予測の差（evaluate.pred_gap_t、取り違えの物差しと同じ量）。
物差しが付けるはずのラベル：A=12・B・C=8 の正確な係数のうち、0.02 以内でいちばん近いもの（なければ「どれにも当たらない」）
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "step2_attention"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import evaluate as E  # noqa: E402

AMP, MEAN = 0.4, 0.5
MATCH = 0.02
A_P, C_P = 12.0, 8.0
B_PS = (24.0, 30.0, 38.0, 48.0)
NOISES = (0.02, 0.025, 0.03)
OUT = ROOT / "runs" / "ls_bias"


def w_ls(P, noise):
    """十分長い材料の最小二乗の解（入力にノイズが乗ったとき）"""
    om = 2 * math.pi / P
    a = AMP ** 2 / 2
    g0, g1, g2 = a + noise ** 2, a * math.cos(om), a * math.cos(2 * om)
    det = g0 * g0 - g1 * g1
    w1 = (g1 * g0 - g1 * g2) / det
    w2 = (g0 * g2 - g1 * g1) / det
    return w1, w2, MEAN * (1 - w1 - w2)


def gap(w, P):
    u = torch.tensor(w, dtype=torch.float32).view(1, 3, 1, 1)
    return float(E.pred_gap_t(u, torch.tensor([P], dtype=torch.float32).view(1, 1, 1))[0, 0, 0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for noise in NOISES:
        for P in (A_P, C_P) + B_PS:
            w = w_ls(P, noise)
            c = math.cos(2 * math.pi / P)
            g_own = gap(w, P)
            pats = {"A=12": A_P, "C=8": C_P} if P in (A_P, C_P) else {"A=12": A_P, f"B={P:g}": P, "C=8": C_P}
            gs = {k: gap(w, p) for k, p in pats.items()}
            best = min(gs, key=gs.get)
            label = best if gs[best] <= MATCH else "どれにも当たらない"
            rows.append(dict(noise=noise, P=P, w_ls=w, exact=(2 * c, -1.0, 1 - c), g_a=g_own, gaps=gs, label=label))
    L = ["# 取り違えの物差しの確かめ (a)：ノイズで学ぶ係数のずれ（式。DESIGN.md 16章）", "",
         "w_LS：十分長い材料の最小二乗の解（入力にノイズ）。g_a：w_LS と正確な係数の、きれいな波の上での1歩先の予測の差（取り違えの物差しと同じ量。線は0.02）", "",
         "| ノイズ | 周期 | w_LS（w₁, w₂, b） | 正確な係数（w₁, w₂, b） | g_a | ほかの模様との差 | 物差しが付けるはずのラベル |", "|---:|---:|---|---|---:|---|---|"]
    for r in rows:
        others = "、".join(f"{k} {v:.3f}" for k, v in r["gaps"].items())
        L.append(f"| {r['noise']:g} | {r['P']:g} | {r['w_ls'][0]:.3f}, {r['w_ls'][1]:.3f}, {r['w_ls'][2]:.3f} | {r['exact'][0]:.3f}, {r['exact'][1]:.3f}, {r['exact'][2]:.3f} | "
                 f"**{r['g_a']:.4f}**{'（線の外）' if r['g_a'] > MATCH else ''} | {others} | {r['label']} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
