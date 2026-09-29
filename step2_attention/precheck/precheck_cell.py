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


def simulate_sigma(P1, P2, sigma, dense=False, n=0.005, smin=0.005, eta=1.0, T=2600, tc=2000, theta=4.0,
                   gR=0.9, eps=1e-3, noise_zone=False, seed=0, init="zero"):
    """
    σ の追いかけ方の2案を比べる模擬（2026-09-29、そうの判断3）。書き換えはどちらも (b)：最小二乗の解へ η で近づける。
    sigma：("uniform", 窓) なら全員一律の窓。("learn", 遅い窓, 速い窓, 直近) なら (c)「学んでいる間はゆっくり」：
      この回より前の「直近」ステップ以内に引き金が引かれていたら遅い窓、そうでなければ速い窓で σ を追いかける
    dense=False：2b のように、引き金（濃度 c > θ かつ不応期でない）が書き換えを決める
    dense=True ：2a のように、全員が毎ステップ書き換える。引き金は同じように数える（濃度を0に戻し、不応期も同じ）が、書き換えは止めない
    返すもの：引き金の有無 F、誤差の2乗 E2、σ S、z² Z（どれも (T, セル)）
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, CELLS)
    s1 = np.full(CELLS, 0.5)
    s2 = np.full(CELLS, 0.5)
    w = np.zeros((CELLS, 3))
    v = np.full(CELLS, smin ** 2)
    if init == "trained":   # 診断用：学び終えた状態（正確な係数、σ＝当てられない誤差、最初の2つの値も波）から始める
        w = np.tile(exact_w(P1), (CELLS, 1))
        v = np.full(CELLS, max(smin, irreducible(n, P1)) ** 2)
        s2 = 0.5 + A * np.sin(phi - 4 * np.pi / P1)
        s1 = 0.5 + A * np.sin(phi - 2 * np.pi / P1)
    R = np.zeros((CELLS, 3, 3))
    r = np.zeros((CELLS, 3))
    c = np.zeros(CELLS)
    ref = np.zeros(CELLS, bool)
    since = np.full(CELLS, 10 ** 9)       # 最後に引き金が引かれてから何ステップか（この回の頭で数える）
    mu = (w * np.stack([s1, s2, np.ones(CELLS)], 1)).sum(1)   # 最初の予測も係数から出す（学び終えた状態から始める診断のため）
    I3 = np.eye(3)
    F = np.zeros((T, CELLS), bool)
    E2 = np.zeros((T, CELLS))
    S = np.zeros((T, CELLS))
    Z = np.zeros((T, CELLS))
    for t in range(T):
        P = P1 if t < tc else P2
        if noise_zone:
            s = rng.uniform(0, 1, CELLS)
        else:
            s = 0.5 + A * np.sin(phi) + n * rng.standard_normal(CELLS)
        phi = phi + 2 * np.pi / P
        f_prev = np.stack([s1, s2, np.ones(CELLS)], 1)
        e = s - mu
        sigma_now = np.maximum(smin, np.sqrt(v))
        z2 = e ** 2 / sigma_now ** 2
        R = gR * R + f_prev[:, :, None] * f_prev[:, None, :]
        r = gR * r + s[:, None] * f_prev
        c = 0.9 * c + 0.1 * z2
        if sigma[0] == "uniform":
            a = np.full(CELLS, 1 / sigma[1])
        else:
            _, slow, fast, recent = sigma
            a = np.where(since <= recent, 1 / slow, 1 / fast)   # この回より前の引き金だけで決める
        v = (1 - a) * v + a * e ** 2
        trig = (c > theta) & ~ref
        write = np.ones(CELLS, bool) if dense else trig
        if write.any():
            wls = np.linalg.solve(R[write] + eps * I3, r[write][:, :, None])[:, :, 0]
            w[write] = w[write] + np.tanh(eta * (wls - w[write]))
        c[trig] = 0
        ref = trig
        since = np.where(trig, 1, since + 1)
        F[t] = trig
        E2[t] = e ** 2
        S[t] = np.maximum(smin, np.sqrt(v))
        Z[t] = z2
        s2, s1 = s1, s
        mu = (w * np.stack([s1, s2, np.ones(CELLS)], 1)).sum(1)
    return F, E2, S, Z


def simulate_window(P1, P2, stop, K=20, delta=0.5, n=0.005, smin=0.005, T=2600, tc=2000, theta=4.0,
                    gR=0.9, eps=1e-3, tau=100, noise_zone=False, seed=0, init="zero", trace=None):
    """
    学習の窓（2026-09-29、そうの判断3）。引き金で窓が開き、窓が開いている間は毎ステップ発火して
    （最小二乗の解を求め、止まる条件を確かめ、止まらなければ書き換える）、止まる条件か上限 K 回で閉じる。
    窓が開いている間は濃度を0に戻さず、閉じたら0に戻して1ステップ休む。σ は一律の窓 tau。
    stop="surprise"：候補3（驚きが収まったら止まる：濃度 c < θ）
    stop="learned" ：学習の窓（学び終えたら止まる：予測の違い d = √((w_ls − w)ᵀ R̄ (w_ls − w)) < delta·σ）
    返すもの：F（発火）、E2、S、Z（どれも (T, セル)）と、閉じた窓の発火の数の一覧・上限に届いたか
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, CELLS)
    s1 = np.full(CELLS, 0.5)
    s2 = np.full(CELLS, 0.5)
    w = np.zeros((CELLS, 3))
    v = np.full(CELLS, smin ** 2)
    if init == "trained":   # 診断用：学び終えた状態（正確な係数、σ＝当てられない誤差、最初の2つの値も波）から始める
        w = np.tile(exact_w(P1), (CELLS, 1))
        v = np.full(CELLS, max(smin, irreducible(n, P1)) ** 2)
        s2 = 0.5 + A * np.sin(phi - 4 * np.pi / P1)
        s1 = 0.5 + A * np.sin(phi - 2 * np.pi / P1)
    R = np.zeros((CELLS, 3, 3))
    r = np.zeros((CELLS, 3))
    mass = np.zeros(CELLS)                 # R の重みの合計（R̄ = R / mass）
    c = np.zeros(CELLS)
    ref = np.zeros(CELLS, bool)
    open_ = np.zeros(CELLS, bool)
    count = np.zeros(CELLS, int)
    mu = (w * np.stack([s1, s2, np.ones(CELLS)], 1)).sum(1)
    I3 = np.eye(3)
    a = 1 / tau
    F = np.zeros((T, CELLS), bool)
    E2 = np.zeros((T, CELLS))
    S = np.zeros((T, CELLS))
    Z = np.zeros((T, CELLS))
    lengths, capped = [], []
    for t in range(T):
        P = P1 if t < tc else P2
        if noise_zone:
            s = rng.uniform(0, 1, CELLS)
        else:
            s = 0.5 + A * np.sin(phi) + n * rng.standard_normal(CELLS)
        phi = phi + 2 * np.pi / P
        f_prev = np.stack([s1, s2, np.ones(CELLS)], 1)
        e = s - mu
        sigma_now = np.maximum(smin, np.sqrt(v))
        z2 = e ** 2 / sigma_now ** 2
        R = gR * R + f_prev[:, :, None] * f_prev[:, None, :]
        r = gR * r + s[:, None] * f_prev
        mass = gR * mass + 1
        c = 0.9 * c + 0.1 * z2
        v = (1 - a) * v + a * e ** 2
        newly = ~open_ & (c > theta) & ~ref
        open_ = open_ | newly
        count[newly] = 0
        fire = open_.copy()
        closing = np.zeros(CELLS, bool)
        if fire.any():
            idx = np.nonzero(fire)[0]
            wls = np.linalg.solve(R[idx] + eps * I3, r[idx][:, :, None])[:, :, 0]
            diff = wls - w[idx]
            count[idx] += 1
            Rbar = R[idx] / mass[idx][:, None, None]
            d = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", diff, Rbar, diff), 0))
            if stop == "surprise":
                done = c[idx] < theta
            else:
                done = d < delta * sigma_now[idx]
            if trace is not None and t < 60:    # 診断用：最初の60ステップの、セル0の様子
                if 0 in idx:
                    k = int(np.nonzero(idx == 0)[0][0])
                    trace.append((t, int(count[0]), float(np.sqrt(e[0] ** 2)), float(sigma_now[0]), float(d[k]),
                                  float(delta * sigma_now[0]), bool(done[k]), float(np.linalg.eigvalsh(Rbar[k])[0])))
            upd = ~done
            w[idx[upd]] = w[idx[upd]] + np.tanh(diff[upd])
            close = done | (count[idx] >= K)
            closing[idx[close]] = True
            lengths.extend(count[idx[close]].tolist())
            capped.extend((~done[close] & (count[idx[close]] >= K)).tolist())
        open_ = open_ & ~closing
        c[closing] = 0
        ref = closing
        F[t] = fire
        E2[t] = e ** 2
        S[t] = np.maximum(smin, np.sqrt(v))
        Z[t] = z2
        s2, s1 = s1, s
        mu = (w * np.stack([s1, s2, np.ones(CELLS)], 1)).sum(1)
    return F, E2, S, Z, np.array(lengths), np.array(capped, bool)


def judge_window(stop, tc=2000, **kw):
    """学習の窓の案1つについて、事前に決めた指標をすべて出す（EXPERIMENTS.md 2026-09-29「学習の窓」）"""
    out = dict(changes={})
    base = None
    for a_, b_ in [(12, 24), (12, 20), (12, 36), (24, 12)]:
        F, E2, S, Z, L, C = simulate_window(a_, b_, stop, tc=tc, **kw)
        pre = np.sqrt(E2[tc - 400:tc].mean(0))
        post = np.sqrt(E2[tc + 250:tc + 300].mean(0))
        out["changes"][f"{a_}→{b_}"] = dict(
            react=F[tc:tc + 10].any(0).mean(), back=(post <= 2 * pre).mean(),
            gaveup=(S[tc + 299] > 2 * S[tc - 1]).mean(), after=F[tc + 200:tc + 300].mean(),
            relearn=F[tc:tc + 300].sum(0).mean())
        if base is None:
            base = dict(
                wave_rate=F[tc - 400:tc].mean(), wave_z2=Z[tc - 400:tc].mean(),
                err_ratio=np.sqrt(E2[tc - 400:tc].mean()) / irreducible(0.005, a_),
                wave_sigma=np.median(S[tc - 1]), early_rate=F[400:800].mean(), early_z2=Z[400:800].mean(),
                t_quiet_wave=settle_time(windowed(F, 50), lambda x: x <= 0.01, tc - 50),
                t_calib_wave=settle_time(windowed(Z, 100), lambda x: (x >= 0.5) & (x <= 1.5), tc - 100),
                learn_fires=F[:tc].sum(0).mean(), win_mean=L.mean() if len(L) else 0.0,
                win_capped=C.mean() if len(C) else 0.0, n_windows=len(L) / CELLS)
    out.update(base)
    F, E2, S, Z, L, C = simulate_window(12, 12, stop, tc=tc, noise_zone=True, **kw)
    out.update(noise_rate=F[tc - 400:tc].mean(), noise_z2=Z[tc - 400:tc].mean(), noise_sigma=np.median(S[tc - 1]),
               noise_learn_fires=F[:tc].sum(0).mean(), noise_win_capped=C.mean() if len(C) else 0.0,
               t_quiet_noise=settle_time(windowed(F, 50), lambda x: x <= 0.01, tc - 50))
    ch = out["changes"].values()
    out["works"] = (all(c_["react"] >= 0.9 and c_["back"] >= 0.95 and c_["gaveup"] <= 0.05 and c_["after"] <= 0.01 for c_ in ch)
                    and out["wave_rate"] <= 0.01 and 0.5 <= out["wave_z2"] <= 1.5
                    and 0.5 <= out["noise_z2"] <= 1.5 and out["noise_rate"] <= 0.01)
    return out


def settle_time(series, ok, end):
    """series[t]（窓の平均）が ok を満たし、それが end まで続く最初の t。最後まで満たさなければ None"""
    good = ok(series[:end])
    if not good[-1]:
        return None
    bad = np.nonzero(~good)[0]
    return 0 if len(bad) == 0 else int(bad[-1] + 1)


def windowed(a, k):
    """(T, セル) を、セルで平均してから k ステップの窓で平均した列（t から t+k−1 まで）"""
    m = a.mean(1)
    return np.convolve(m, np.ones(k) / k, mode="valid")


def judge_sigma(sigma, dense=False, tc=2000):
    """σ の案1つについて、事前に決めた指標をすべて出す（EXPERIMENTS.md 2026-09-29「σ の2案の比べ方」）"""
    out = dict(changes={})
    base = None
    for a, b in [(12, 24), (12, 20), (12, 36), (24, 12)]:
        F, E2, S, Z = simulate_sigma(a, b, sigma, dense=dense, tc=tc)
        pre = np.sqrt(E2[tc - 400:tc].mean(0))
        post = np.sqrt(E2[tc + 250:tc + 300].mean(0))
        out["changes"][f"{a}→{b}"] = dict(
            react=F[tc:tc + 10].any(0).mean(), back=(post <= 2 * pre).mean(),
            gaveup=(S[tc + 299] > 2 * S[tc - 1]).mean(), after=F[tc + 200:tc + 300].mean(),
            relearn=F[tc:tc + 300].sum(0).mean())
        if base is None:      # 変える前の落ち着いた様子と、落ち着くまでの時間は 12→24 の回で測る（変える前は同じ世界）
            base = dict(
                wave_rate=F[tc - 400:tc].mean(), wave_z2=Z[tc - 400:tc].mean(),
                err_ratio=np.sqrt(E2[tc - 400:tc].mean()) / irreducible(0.005, a),
                wave_sigma=np.median(S[tc - 1]),
                early_rate=F[400:800].mean(), early_z2=Z[400:800].mean(),
                early_err=np.sqrt(E2[600:800].mean()) / irreducible(0.005, a),
                t_quiet_wave=settle_time(windowed(F, 50), lambda x: x <= 0.01, tc - 50),
                t_calib_wave=settle_time(windowed(Z, 100), lambda x: (x >= 0.5) & (x <= 1.5), tc - 100),
                learn_fires=F[:400].sum(0).mean())
    out.update(base)
    F, E2, S, Z = simulate_sigma(12, 12, sigma, dense=dense, tc=tc, noise_zone=True)
    out.update(noise_rate=F[tc - 400:tc].mean(), noise_z2=Z[tc - 400:tc].mean(), noise_sigma=np.median(S[tc - 1]),
               noise_early_rate=F[400:800].mean(),
               t_quiet_noise=settle_time(windowed(F, 50), lambda x: x <= 0.01, tc - 50))
    ch = out["changes"].values()
    out["works"] = (all(c["react"] >= 0.9 and c["back"] >= 0.95 and c["gaveup"] <= 0.05 and c["after"] <= 0.01 for c in ch)
                    and out["wave_rate"] <= 0.01 and 0.5 <= out["wave_z2"] <= 1.5
                    and 0.5 <= out["noise_z2"] <= 1.5 and out["noise_rate"] <= 0.01)
    ts = [out["t_quiet_noise"], out["t_quiet_wave"], out["t_calib_wave"]]
    out["settle"] = None if any(x is None for x in ts) else max(ts)
    return out


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

    # ---- 2026-09-29（続き）そうの判断のあと：学習に混ぜる変化の境目と、σ の追いかけ方の2案 ----
    out += ["", "## 表6：変えた直後の z² の平均と |cos ω の差|（ノイズ0.005、学び終えた予測器）", "",
            "学習に混ぜる周期の変化の境目を決めるため。理屈の近似は 1 ＋ 2·0.4²·(差)² ÷ 当てられない誤差²。", "",
            "| 変化 | \\|cos ω の差\\| | z² の平均（測った） | 理屈の近似 |", "|---|---:|---:|---:|"]
    vis_pairs = []
    for P1 in (12.0, 20.0):
        for d in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.10):
            for sign in (+1, -1):
                cc = np.cos(2 * np.pi / P1) + sign * d
                if cc >= 1 or cc <= np.cos(2 * np.pi / 8):
                    continue
                vis_pairs.append((P1, 2 * np.pi / np.arccos(cc)))
    for P1, P2 in vis_pairs:
        dc = abs(np.cos(2 * np.pi / P2) - np.cos(2 * np.pi / P1))
        approx = 1 + 2 * A ** 2 * dc ** 2 / irreducible(0.005, P1) ** 2
        out.append(f"| {P1:.0f}→{P2:.1f} | {dc:.3f} | {visibility(0.005, 0.005, P1, P2):.1f} | {approx:.1f} |")

    arms = [("参照：窓100（今の設計）", ("uniform", 100)),
            ("(d) 一律 窓150", ("uniform", 150)), ("(d) 一律 窓300", ("uniform", 300)), ("(d) 一律 窓600", ("uniform", 600)),
            ("(c) 学んでいる間はゆっくり（1000／100、直近10）", ("learn", 1000, 100, 10))]
    res = {name: judge_sigma(sig) for name, sig in arms}
    res_dense = {name: judge_sigma(sig, dense=True) for name, sig in arms}

    def fmt_t(x):
        return "—（届かず）" if x is None else f"{x}"

    out += ["", "## 表7：σ の追いかけ方の2案（2b のように、引き金が書き換えを決める）。周期の変化ごと", "",
            "書き換えは (b)（最小二乗の解へ η=1 で近づける）、ノイズ0.005、θ=4、予測器0から。0〜2000 で落ち着き、2000 で周期を変える。", "",
            "| 案 | 変化 | 10歩以内の反応 | 誤差が戻った | あきらめ | 学び直し後の発火率 | 学び直しの発火/セル |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for name, o in res.items():
        for ch, cm in o["changes"].items():
            out.append(f"| {name} | {ch} | {cm['react']:.2f} | {cm['back']:.2f} | {cm['gaveup']:.2f} | {cm['after']:.4f} | {cm['relearn']:.1f} |")

    def steady_table(rs, title):
        rows = ["", title, "",
                "| 案 | 最初の400の発火/セル | 波の発火率 | 波の z² | 誤差÷当てられない誤差 | 波の σ | ノイズの発火率 | ノイズの z² | ノイズの σ | "
                "400〜800 の波の発火率・z² | 400〜800 のノイズの発火率 | 効く |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
        for name, o in rs.items():
            rows.append(f"| {name} | {o['learn_fires']:.1f} | {o['wave_rate']:.4f} | {o['wave_z2']:.2f} | {o['err_ratio']:.2f} | {o['wave_sigma']:.4f} | "
                        f"{o['noise_rate']:.4f} | {o['noise_z2']:.2f} | {o['noise_sigma']:.3f} | "
                        f"{o['early_rate']:.4f}・{o['early_z2']:.2f} | {o['noise_early_rate']:.4f} | {'○' if o['works'] else '×'} |")
        return rows

    out += steady_table(res, "## 表8：変える前の落ち着いた様子（1600〜2000）と、もとのプロトコルの目安（400〜800）。2b のように")
    out += ["", "## 表9：落ち着くまでの時間（0から数えたステップ。2b のように）", "",
            "| 案 | ノイズの区域が静かになる | 波が静かになる | 波の自信度が合う | 一番長いもの | 効く |", "|---|---:|---:|---:|---:|---|"]
    for name, o in res.items():
        out.append(f"| {name} | {fmt_t(o['t_quiet_noise'])} | {fmt_t(o['t_quiet_wave'])} | {fmt_t(o['t_calib_wave'])} | "
                   f"{fmt_t(o['settle'])} | {'○' if o['works'] else '×'} |")
    out += steady_table(res_dense, "## 表9b：2a のように（全員が毎ステップ書き換える。引き金は数えるだけ）。確かめとして")
    out += ["", "| 案（2a のように） | ノイズの区域が静かになる | 波が静かになる | 波の自信度が合う |", "|---|---:|---:|---:|"]
    for name, o in res_dense.items():
        out.append(f"| {name} | {fmt_t(o['t_quiet_noise'])} | {fmt_t(o['t_quiet_wave'])} | {fmt_t(o['t_calib_wave'])} |")

    # 事前に決めた決め方（EXPERIMENTS.md）
    c_name = arms[-1][0]
    d_names = [nm for nm, sig in arms[1:4]]
    c_ok = res[c_name]["works"]
    d_ok = [nm for nm in d_names if res[nm]["works"] and res[nm]["settle"] is not None]
    if c_ok and d_ok:
        best_d = min(d_ok, key=lambda nm: res[nm]["settle"])
        cs = res[c_name]["settle"]
        if cs is not None and res[best_d]["settle"] <= 2 * cs:
            verdict = f"両方「効く」、(d) の落ち着くまで {res[best_d]['settle']} ≤ (c) の2倍（{2 * cs}）→ **(d) {best_d}**（単純なほう）"
        else:
            verdict = f"両方「効く」が、(d) の落ち着くまで {res[best_d]['settle']} が (c) の2倍（{None if cs is None else 2 * cs}）を超える → **(c)**（数字をそうに見せる）"
    elif c_ok:
        verdict = "(c) だけ「効く」 → **(c)**"
    elif d_ok:
        verdict = f"(d) だけ「効く」 → **(d) {min(d_ok, key=lambda nm: res[nm]['settle'])}**"
    else:
        verdict = "どちらも「効かない」 → **止めて報告する**"
    out += ["", f"**事前に決めた決め方による判定**：{verdict}"]

    # 2a のように動かしたときの、周期の変化ごとの数字（確かめ）
    out += ["", "## 表9c：2a のように動かしたときの、周期の変化ごとの数字（引き金は数えるだけ。確かめ）", "",
            "| 案 | 変化 | 10歩以内の引き金 | 誤差が戻った | あきらめ | 学び直し後の引き金の率 |", "|---|---|---:|---:|---:|---:|"]
    for name, o in res_dense.items():
        for ch, cm in o["changes"].items():
            out.append(f"| {name} | {ch} | {cm['react']:.2f} | {cm['back']:.2f} | {cm['gaveup']:.2f} | {cm['after']:.4f} |")

    # 診断（判定には使わない）：学び終えた状態から始めたら、学び直しはどうか。「効かない」の原因が最初の学びだけかを切り分ける
    out += ["", "## 表10（診断。判定には使わない）：学び終えた状態から始める（2b のように、400 で周期を変える）", "",
            "予測器0から始める最初の学び（表8で誤差が3.8〜12倍のまま）を外して、驚きでの学び直しだけを見る。", "",
            "| 案 | 変化 | 変える前の発火率 | 変える前の z² | 10歩以内の反応 | 誤差が戻った | あきらめ | 学び直し後の発火率 | 学び直しの発火/セル |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, sig in arms:
        for a, b in [(12, 24), (12, 20), (12, 36), (24, 12)]:
            F, E2, S, Z = simulate_sigma(a, b, sig, T=1000, tc=400, init="trained")
            pre = np.sqrt(E2[200:400].mean(0))
            post = np.sqrt(E2[650:700].mean(0))
            out.append(f"| {name} | {a}→{b} | {F[200:400].mean():.4f} | {Z[200:400].mean():.2f} | {F[400:410].any(0).mean():.2f} | "
                       f"{(post <= 2 * pre).mean():.2f} | {(S[699] > 2 * S[399]).mean():.2f} | {F[600:700].mean():.4f} | "
                       f"{F[400:700].sum(0).mean():.1f} |")

    # ---- 2026-09-29（続き）学習の窓（そうの判断3）：候補3と、止まる条件の1点だけを変えて比べる ----
    wins = [("候補3「驚きが収まったら止まる」（濃度 < θ）", "surprise"), ("学習の窓「学び終えたら止まる」（予測の違い < 0.5σ）", "learned")]
    wres = {name: judge_window(stop) for name, stop in wins}
    ref_name, ref = arms[0][0], res[arms[0][0]]
    out += ["", "## 表11：学習の窓と候補3（止まる条件だけが違う。σ は窓100、1回の窓の発火は20回まで）。周期の変化ごと", "",
            "| 案 | 変化 | 10歩以内の反応 | 誤差が戻った | あきらめ | 学び直し後の発火率 | 学び直しの発火/セル |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for name, o in [(f"参照：窓なし（{ref_name}）", ref)] + list(wres.items()):
        for ch, cm in o["changes"].items():
            out.append(f"| {name} | {ch} | {cm['react']:.2f} | {cm['back']:.2f} | {cm['gaveup']:.2f} | {cm['after']:.4f} | {cm['relearn']:.1f} |")
    out += ["", "## 表12：学習の窓と候補3：変える前の落ち着いた様子（1600〜2000）、学び始めの費用、窓の長さ、落ち着くまでの時間", "",
            "| 案 | 0〜2000 の発火/セル（波） | 1回の窓の発火（平均） | 上限に届いた窓 | 波の発火率 | 波の z² | 誤差÷当てられない誤差 | "
            "ノイズの発火率 | ノイズの z² | ノイズの σ | ノイズ 0〜2000 の発火/セル | ノイズが静かになる | 波が静かになる | 波の自信度が合う | 効く |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, o in wres.items():
        out.append(f"| {name} | {o['learn_fires']:.1f} | {o['win_mean']:.1f} | {o['win_capped']:.2f} | {o['wave_rate']:.4f} | {o['wave_z2']:.2f} | "
                   f"{o['err_ratio']:.2f} | {o['noise_rate']:.4f} | {o['noise_z2']:.2f} | {o['noise_sigma']:.3f} | {o['noise_learn_fires']:.1f} | "
                   f"{fmt_t(o['t_quiet_noise'])} | {fmt_t(o['t_quiet_wave'])} | {fmt_t(o['t_calib_wave'])} | {'○' if o['works'] else '×'} |")
    # 診断（判定には使わない）：学び終えた状態から始めたら、学び直しはどうか（最初の学びを外して、止まる条件の違いだけを見る）
    out += ["", "## 表13（診断。判定には使わない）：学習の窓と候補3を、学び終えた状態から始める（400 で周期を変える）", "",
            "| 案 | 変化 | 変える前の発火率 | 10歩以内の反応 | 誤差が戻った | あきらめ | 学び直し後の発火率 | 学び直しの発火/セル | 上限に届いた窓 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, stop in wins:
        for a_, b_ in [(12, 24), (12, 20), (12, 36), (24, 12)]:
            F, E2, S, Z, L, C = simulate_window(a_, b_, stop, T=1000, tc=400, init="trained")
            pre = np.sqrt(E2[200:400].mean(0))
            post = np.sqrt(E2[650:700].mean(0))
            out.append(f"| {name} | {a_}→{b_} | {F[200:400].mean():.4f} | {F[400:410].any(0).mean():.2f} | {(post <= 2 * pre).mean():.2f} | "
                       f"{(S[699] > 2 * S[399]).mean():.2f} | {F[600:700].mean():.4f} | {F[400:700].sum(0).mean():.1f} | "
                       f"{(C.mean() if len(C) else 0):.2f} |")
    # 診断：学習の窓の最初の学び（予測器0から）で、窓がいつ・なぜ閉じたか（セル0）
    tr = []
    simulate_window(12, 24, "learned", T=80, tc=2000, trace=tr)
    out += ["", "## 表14（診断）：学習の窓の最初の学び（予測器0から、12の波、セル0）。窓が開いている間の毎ステップ", "",
            "| ステップ | 窓の中の何回目 | 誤差の大きさ | σ | 予測の違い d | 止まる境目 0.5σ | 止まった | R̄ の一番小さい固有値 |",
            "|---:|---:|---:|---:|---:|---:|---|---:|"]
    for t_, k_, err_, sig_, d_, thr_, dn_, ev_ in tr:
        out.append(f"| {t_} | {k_} | {err_:.4f} | {sig_:.4f} | {d_:.4f} | {thr_:.4f} | {'○' if dn_ else ''} | {ev_:.2e} |")
    lw, c3 = wres[wins[1][0]], wres[wins[0][0]]
    if lw["works"]:
        wv = "学習の窓が「効く」→ **学習の窓を採る**（仕組みの1つ目）" + ("。候補3も「効く」" if c3["works"] else "。候補3は「効かない」")
    elif c3["works"]:
        wv = "候補3だけ「効く」→ **そうに報告して決めてもらう**"
    else:
        wv = "どちらも「効かない」→ **止めて報告する**"
    out += ["", f"**事前に決めた決め方による判定（学習の窓）**：{wv}"]

    out += ["", f"（計算 {time.time() - t0:.0f} 秒）", ""]
    (HERE / "results.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
