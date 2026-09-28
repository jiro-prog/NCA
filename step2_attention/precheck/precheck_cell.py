"""
Step 2 実装前の模擬（1セル、DNA なし）。2026-09-29。

設計書（DESIGN.md）の 2b・2c の仕組み（z² の引き金、σ の移動平均＝案 (B)、痕跡つきデルタ則）を、
セル 256 個を独立に numpy で回して、「驚き → 発火 → 学び直し → 静けさ」が起きるかを、実装の前に確かめる。
DNA は入れない（手で決めた書き換え＝「DNA なし」の対照にあたる）。近所もない（1セルずつ独立）。

    .venv/Scripts/python.exe step2_attention/precheck/precheck_cell.py

結果は同じフォルダの results.md に書く（シード固定）。

表1：周期を変えたときの驚きの大きさ（学び終えた予測器、z² の平均）。世界のノイズ 0.02 と 0.005
表2：設計どおり（ノイズ 0.02、予測器0から、σ の窓 約100、θ=4、痕跡つきデルタ則）。最初の学習で「あきらめる」か
表3：設計書の逃げ道「σ の窓を長くする」だけで足りるか（窓 約333・約1000）
表4：θ を下げたら（θ=2）、ノイズ 0.02 のままで周期の変化に反応するか。学び終えた状態から始めて、空振りの発火も測る
表5：直し方の候補（発火のときに最近の窓の最小二乗の解へ近づける＋σ は上がるのだけ遅い＋ノイズ 0.005）
"""
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = 0.4          # 波の振幅（s = 0.5 ± 0.4）
TC = 800         # 周期を変える時刻
T = 1400         # 模擬の長さ
CELLS = 256


def irreducible(n, P):
    """2つ前までの線形予測で当てられない誤差（センサーのノイズ n が入力にも乗る分）"""
    return n * np.sqrt(2 + 4 * np.cos(2 * np.pi / P) ** 2)


def exact_w(P):
    """ノイズなしの正弦波を正確に当てる係数：s(t+1) = 2cos(ω)s(t) − s(t−1) + (1 − cos ω)（平均0.5）"""
    c = np.cos(2 * np.pi / P)
    return np.array([2 * c, -1.0, 1 - c])


def simulate(n, smin, P1, P2, eta, learner="delta", init="zero", tau_up=100, tau_down=100, theta=4.0,
             gR=0.9, eps=1e-3, noise_zone=False, seed=0):
    """
    毎ステップ（全セル、固定の式）：誤差 e と z²、痕跡 tr ← 0.9·tr ＋ e·f、濃度 c ← 0.9·c ＋ 0.1·z²、
      σ の移動平均 v（誤差が大きいときは 1/tau_up、小さいときは 1/tau_down の速さで追いかける。同じなら設計どおりの案 (B)）、
      最小二乗の統計 R ← gR·R ＋ f fᵀ、r ← gR·r ＋ s·f（learner="ls" のときだけ使う）
    発火（c > θ かつ不応期でない）したとき：
      delta：w ← w ＋ tanh(η·tr)            （設計どおり。g=1、Δw=0）
      ls   ：w ← w ＋ tanh(η·(w_ls − w))、w_ls = (R＋εI)⁻¹ r（最近の窓でデルタ則を収束まで回した結果）
      そのあと痕跡と濃度を0に戻し、1ステップ休む
    init："zero"（設計どおり、予測器0・σ は下限から）か "trained"（周期 P1 の正確な係数・σ は当てられない誤差から）
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, CELLS)
    s1 = np.full(CELLS, 0.5)
    s2 = np.full(CELLS, 0.5)
    if init == "trained":
        w = np.tile(exact_w(P1), (CELLS, 1))
        v = np.full(CELLS, max(smin, irreducible(n, P1)) ** 2)
        # 最初の2つの値も波にそろえる（学び終えた状態から始める）
        s2 = 0.5 + A * np.sin(phi - 4 * np.pi / P1)
        s1 = 0.5 + A * np.sin(phi - 2 * np.pi / P1)
    else:
        w = np.zeros((CELLS, 3))
        v = np.full(CELLS, smin ** 2)
    tr = np.zeros((CELLS, 3))
    R = np.zeros((CELLS, 3, 3))
    r = np.zeros((CELLS, 3))
    c = np.zeros(CELLS)
    ref = np.zeros(CELLS, bool)
    mu = (w * np.stack([s1, s2, np.ones(CELLS)], 1)).sum(1)
    I3 = np.eye(3)
    au, ad = 1 / tau_up, 1 / tau_down
    F = np.zeros((T, CELLS), bool)
    E2 = np.zeros((T, CELLS))
    S = np.zeros((T, CELLS))
    Z = np.zeros((T, CELLS))
    for t in range(T):
        P = P1 if t < TC else P2
        if noise_zone:
            s = rng.uniform(0, 1, CELLS)
        else:
            s = 0.5 + A * np.sin(phi) + n * rng.standard_normal(CELLS)
        phi = phi + 2 * np.pi / P  # 位相はつなげたまま、周期だけ変える
        f_prev = np.stack([s1, s2, np.ones(CELLS)], 1)
        e = s - mu
        sigma = np.maximum(smin, np.sqrt(v))
        z2 = e ** 2 / sigma ** 2
        tr = 0.9 * tr + e[:, None] * f_prev
        if learner == "ls":
            R = gR * R + f_prev[:, :, None] * f_prev[:, None, :]
            r = gR * r + s[:, None] * f_prev
        c = 0.9 * c + 0.1 * z2
        a = np.where(e ** 2 > v, au, ad)
        v = (1 - a) * v + a * e ** 2
        fire = (c > theta) & ~ref
        if fire.any():
            if learner == "ls":
                wls = np.linalg.solve(R[fire] + eps * I3, r[fire][:, :, None])[:, :, 0]
                w[fire] = w[fire] + np.tanh(eta * (wls - w[fire]))
            else:
                w[fire] = w[fire] + np.tanh(eta * tr[fire])
            tr[fire] = 0
            c[fire] = 0
        ref = fire
        F[t] = fire
        E2[t] = e ** 2
        S[t] = np.maximum(smin, np.sqrt(v))
        Z[t] = z2
        s2, s1 = s1, s
        mu = (w * np.stack([s1, s2, np.ones(CELLS)], 1)).sum(1)
    return F, E2, S, Z


def summarize(F, E2, S, Z, n, P1):
    """指標（波の区域）。「変える前」は 600〜800、「学び直し」は 800〜1100"""
    pre = np.sqrt(E2[600:TC].mean(0))
    post = np.sqrt(E2[TC + 250:TC + 300].mean(0))
    return dict(
        learn_fires=F[:400].sum(0).mean(),                     # 最初の400ステップの発火（1セルあたり）
        quiet=F[400:TC].mean(),                                # 落ち着いた時期の発火率
        z2=Z[400:TC].mean(),                                   # 落ち着いた時期の z² の平均（自信度が合っていれば1）
        err_ratio=pre.mean() / irreducible(n, P1),             # 変える前の誤差 ÷ 当てられない誤差
        react=F[TC:TC + 10].any(0).mean(),                     # 変えてから10ステップ以内に発火したセルの割合
        relearn=F[TC:TC + 300].sum(0).mean(),                  # 学び直しの発火（300ステップ、1セルあたり）
        back=(post <= 2 * pre).mean(),                         # 300ステップ後に、誤差が変える前の2倍以内に戻ったセルの割合
        gaveup=(S[TC + 299] > 2 * S[TC - 1]).mean(),           # σ が変える前の2倍より高いまま（あきらめ）
        after=F[TC + 200:TC + 300].mean(),                     # 学び直したあとの発火率
    )


def visibility(n, smin, P1, P2, seed=0):
    """学び終えた予測器（ノイズつきの最小二乗）で、周期を変えた直後の z² の平均"""
    rng = np.random.default_rng(seed)

    def wave(P, T_):
        t = np.arange(T_)
        return 0.5 + A * np.sin(2 * np.pi * t / P) + rng.normal(0, n, T_)

    def feats(s):
        return np.stack([s[1:-1], s[:-2], np.ones(len(s) - 2)], 1), s[2:]

    X, y = feats(wave(P1, 4000))
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    sig = max(smin, (y - X @ w).std())
    X2, y2 = feats(wave(P2, 4000))
    return np.mean((y2 - X2 @ w) ** 2) / sig ** 2


def row(label, o):
    return (f"| {label} | {o['learn_fires']:.1f} | {o['quiet']:.4f} | {o['z2']:.2f} | {o['err_ratio']:.1f} | "
            f"{o['react']:.2f} | {o['relearn']:.1f} | {o['back']:.2f} | {o['gaveup']:.2f} | {o['after']:.4f} |")


HEAD = ("| 設定 | 最初の400の発火/セル | 落ち着いた発火率 | 落ち着いた z² | 誤差÷当てられない誤差 | "
        "10歩以内の反応 | 学び直しの発火/セル | 誤差が戻った | あきらめ | 学び直し後の発火率 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def noise_row(label, n, smin, eta, **kw):
    F, E2, S, Z = simulate(n, smin, 12, 12, eta, noise_zone=True, **kw)
    return f"| {label} | {F[:400].sum(0).mean():.1f} | {F[400:TC].mean():.4f} | {np.median(S[TC - 1]):.3f} | {Z[400:TC].mean():.2f} |"


NOISE_HEAD = ("| 設定 | 最初の400の発火/セル | 落ち着いた発火率 | 800での σ（中央値） | 落ち着いた z² |\n"
              "|---|---:|---:|---:|---:|")


def main():
    t0 = time.time()
    out = ["# Step 2 実装前の模擬（1セル、DNA なし）の結果", "",
           "`precheck_cell.py` の出力（シード固定、セル256個を独立に）。指標の意味はスクリプトの `summarize` を参照。",
           "当てられない誤差：ノイズ0.02なら約0.045、0.005なら約0.011（周期12〜24）。"]
    pairs = [(12, 24), (12, 20), (12, 36), (24, 36), (24, 12)]

    out += ["", "## 表1：周期を変えたときの驚きの大きさ（学び終えた予測器、変えた直後の z² の平均。θ=4 を超えないと反応しない）", "",
            "| 世界のノイズ（σ の下限） | " + " | ".join(f"{a}→{b}" for a, b in pairs) + " |",
            "|---|" + "---:|" * len(pairs)]
    for n, smin in [(0.02, 0.02), (0.005, 0.005)]:
        out.append(f"| {n}（{smin}） | " + " | ".join(f"{visibility(n, smin, a, b):.1f}" for a, b in pairs) + " |")

    out += ["", "## 表2：設計どおり（ノイズ0.02、予測器0から、σ の窓 約100、θ=4、痕跡つきデルタ則）。周期 12→24", "", HEAD]
    for eta in [0.01, 0.03, 0.1, 0.3, 1.0]:
        out.append(row(f"η={eta}", summarize(*simulate(0.02, 0.02, 12, 24, eta), 0.02, 12)))
    out += ["", "ノイズの区域（一様乱数）：", "", NOISE_HEAD, noise_row("η=0.1", 0.02, 0.02, 0.1)]

    out += ["", "## 表3：σ の窓を長くするだけ（ノイズ0.02、予測器0から、θ=4、痕跡つきデルタ則）。周期 12→24", "", HEAD]
    for tau in [333, 1000]:
        for eta in [0.1, 0.3]:
            out.append(row(f"窓 約{tau}、η={eta}", summarize(*simulate(0.02, 0.02, 12, 24, eta, tau_up=tau, tau_down=tau), 0.02, 12)))

    out += ["", "## 表4：θ を下げる（ノイズ0.02、σ の窓 約100）。学び終えた状態（正確な係数、σ＝当てられない誤差）から始める", "",
            "落ち着いた発火率（400〜800）が、θ を下げたときの空振りの発火。", "", HEAD]
    for theta in [4.0, 2.0]:
        for learner, eta in [("delta", 0.1), ("delta", 0.3), ("ls", 1.0)]:
            for a, b in [(12, 24), (12, 20), (24, 36)]:
                o = summarize(*simulate(0.02, 0.02, a, b, eta, learner=learner, init="trained", theta=theta), 0.02, a)
                out.append(row(f"θ={theta:g}、{learner} η={eta}、{a}→{b}", o))

    out += ["", "## 表4b：ノイズ0.005 で、学び終えた状態から始める（σ の窓 約100 の対称＝設計どおりの案 (B)、θ=4）", "",
            "最初の学習（0から）の「あきらめ」を外して、学び直しだけを見る。", "", HEAD]
    for learner, eta in [("delta", 0.1), ("delta", 0.3), ("ls", 1.0)]:
        for a, b in [(12, 24), (12, 20), (12, 36), (24, 12), (24, 36)]:
            o = summarize(*simulate(0.005, 0.005, a, b, eta, learner=learner, init="trained"), 0.005, a)
            out.append(row(f"{learner} η={eta}、{a}→{b}", o))

    out += ["", "## 表5：直し方の候補（ノイズ0.005、σ の下限0.005、予測器0から、θ=4、発火で最小二乗の解へ近づける η=1）", "",
            "σ は「誤差が大きいときは遅く上がり（窓 約1000 か 約300）、小さいときは約100で下がる」。", "", HEAD]
    for tu in [1000, 300]:
        for a, b in [(12, 24), (12, 36), (24, 36), (24, 12)]:
            o = summarize(*simulate(0.005, 0.005, a, b, 1.0, learner="ls", tau_up=tu, tau_down=100), 0.005, a)
            out.append(row(f"上がる窓 約{tu}、{a}→{b}", o))
    out += ["", "比べる：同じ世界で、設計どおりの書き換え（痕跡つきデルタ則 η=0.1、σ の窓 約100）", "", HEAD,
            row("デルタ則、12→24", summarize(*simulate(0.005, 0.005, 12, 24, 0.1), 0.005, 12)),
            row("最小二乗、σ 対称 約100、12→24", summarize(*simulate(0.005, 0.005, 12, 24, 1.0, learner="ls"), 0.005, 12))]
    out += ["", "ノイズの区域（一様乱数）：", "", NOISE_HEAD]
    for tu in [1000, 300]:
        out.append(noise_row(f"最小二乗、上がる窓 約{tu}", 0.005, 0.005, 1.0, learner="ls", tau_up=tu, tau_down=100))

    out += ["", f"（計算 {time.time() - t0:.0f} 秒）", ""]
    (HERE / "results.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
