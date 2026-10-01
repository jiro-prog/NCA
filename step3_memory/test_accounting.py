"""
Step 3 のセル（cell.Cell3）の会計の約束を確かめるテスト（DESIGN.md 10.4、CLAUDE.md の計算の会計）。
回帰の確かめ（check_cell.py：本物のセルと模擬が1ビットも同じ）のあとに回す（2026-10-01、そうの判断の順番）。

    .venv/Scripts/python.exe step3_memory/test_accounting.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step2_attention"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import cell as C  # noqa: E402
import world as W  # noqa: E402
from model import AttentionCell, Physio2  # noqa: E402
from model import init_state as init_state2  # noqa: E402

H = Wd = 12


def world(P=12.0, noise=0.02, n=2, seed=0, zone=None):
    cfg = W.WorldConfig(size=H, zone=H if zone is None else zone, noise=noise)
    g = torch.Generator().manual_seed(seed)
    return cfg, g, W.reset(cfg, P, n=n, generator=g)


def run(cell, T, P=12.0, P2=None, t_change=None, n=2, seed=0, zone=None, st=None):
    """周期 P の世界を T ステップ回す（t_change で P2 に変える）。記録のリストを返す"""
    cfg, g, ws = world(P, n=n, seed=seed, zone=zone)
    st = C.init_state(n, H, Wd) if st is None else st
    recs, states = [], []
    for t in range(T):
        if t_change is not None and t == t_change - 1:
            ws = W.set_period(ws, P2)
        st, rec = cell.step(st, W.sense(ws, cfg, g))
        recs.append(rec)
        states.append(st)
        ws = W.advance(ws)
    return recs, states


def test_floor_is_the_frozen_recipe_bit_for_bit():
    """床（sleep=False）は、Step 2 の凍結したレシピ（AttentionCell(dna=False)、元の設計の引き金）と、状態のすべてが1ビットも同じ"""
    cfg, g, ws = world(seed=1)
    cfg2, g2, ws2 = world(seed=1)
    floor = C.Cell3(sleep=False)
    ref = AttentionCell(dna=False)
    s3 = C.init_state(2, H, Wd)
    s2 = init_state2(2, H, Wd)
    with torch.no_grad():
        for t in range(600):
            if t == 299:
                ws = W.set_period(ws, 24.0)
                ws2 = W.set_period(ws2, 24.0)
            s3, _ = floor.step(s3, W.sense(ws, cfg, g))
            s2, _ = ref.step(s2, W.sense(ws2, cfg2, g2), policy={"kind": "trigger"})
            ws, ws2 = W.advance(ws), W.advance(ws2)
    for a, b in zip(s3.core, s2):
        assert torch.equal(a, b)
    assert not s3.q.any() and not s3.k.any() and not s3.RL.any()      # 床は睡眠の状態を使わない


def test_cells_that_neither_fire_nor_sleep_keep_their_coefficients():
    """発火の会計：驚いて発火もせず、寝てもいないセルは、予測器の係数を1ビットも変えない"""
    recs, states = run(C.Cell3(sleep=True), 800, P=12.0, P2=24.0, t_change=400)
    for t in range(1, len(states)):
        quiet = ~(recs[t]["surprise"] | recs[t]["sleep"])
        same = (states[t].core.w == states[t - 1].core.w).all(1, keepdim=True)
        assert bool(same[quiet].all())


def test_sleep_writes_only_at_doubling_times_after_last_fire():
    """睡眠の書き込みは、最後の驚きの発火から100・300・700・1500…ステップ目にだけ起きる（間隔を2倍ずつ、上限なし）"""
    recs, _ = run(C.Cell3(sleep=True), 3200, P=12.0, P2=24.0, t_change=1000, n=1)
    S = torch.stack([r["sleep"][0, 0] for r in recs])          # (T,H,W)
    F = torch.stack([r["surprise"][0, 0] for r in recs])
    allowed = {100 * (2 ** (k + 1) - 1) for k in range(10)}
    for y in range(H):
        for x in range(Wd):
            last = -1                                           # 始まりは「最後の発火」が t=−1 にあったのと同じ（q は t=0 で1）
            for t in range(S.shape[0]):
                if F[t, y, x]:
                    last = t
                if S[t, y, x]:
                    assert (t - last) in allowed, (y, x, t, last)
    assert S.any()                                              # 実際に寝ている


def test_sleep_and_surprise_never_coincide():
    """驚きの発火と睡眠の書き込みは同じステップに重ならない（別枠で数える）"""
    recs, _ = run(C.Cell3(sleep=True), 1600, P=12.0, P2=24.0, t_change=800)
    for r in recs:
        assert not bool((r["surprise"] & r["sleep"]).any())


def test_long_trace_resets_on_surprise():
    """見直し用の長い痕跡と、静かさの数え・見直しの回数は、驚いて発火したセルで0に戻る"""
    recs, states = run(C.Cell3(sleep=True), 900, P=12.0, P2=24.0, t_change=400)
    seen = False
    for r, s in zip(recs, states):
        f = r["surprise"]
        if f.any():
            seen = True
            assert bool((s.RL.abs().sum(1, keepdim=True)[f] == 0).all())
            assert bool((s.rL.abs().sum(1, keepdim=True)[f] == 0).all())
            assert bool((s.q[f] == 0).all()) and bool((s.k[f] == 0).all())
    assert seen


def test_long_trace_is_the_cheap_sum_every_step():
    """長い痕跡は、驚いて発火していないセルでは毎ステップ γ_L 倍して足すだけ（安い式）"""
    cell = C.Cell3(sleep=True)
    cfg, g, ws = world(seed=3)
    st = C.init_state(2, H, Wd)
    with torch.no_grad():
        for _ in range(50):
            sense = W.sense(ws, cfg, g)
            new, rec = cell.step(st, sense)
            f1, f2 = new.core.hist[:, 1:2], new.core.hist[:, 2:3]
            one = torch.ones_like(f1)
            exp_RL = C.Physio3.gamma_L * st.RL + torch.cat([f1 * f1, f1 * f2, f1 * one, f2 * f2, f2 * one, one], 1)
            keep = ~rec["surprise"]
            assert torch.equal(torch.where(keep, new.RL, exp_RL), exp_RL)
            st = new
            ws = W.advance(ws)


def test_sleep_is_counted_in_noise_zone_too():
    """ノイズの区域のセルも同じに数える（全セル）：ノイズの区域のセルも、最後の驚きの発火から100・300…ステップ目に寝て、睡眠の書き込みに数えられる"""
    recs, _ = run(C.Cell3(sleep=True), 800, zone=6, n=1)
    zone = W.zone_mask(W.WorldConfig(size=H, zone=6), "cpu")
    S = torch.stack([r["sleep"][0, 0] for r in recs])
    F = torch.stack([r["surprise"][0, 0] for r in recs])
    assert bool(S[:, zone].any())                              # ノイズの区域でも実際に寝ている
    allowed = {100 * (2 ** (k + 1) - 1) for k in range(10)}
    for y, x in zone.nonzero().tolist():
        last = -1
        for t in range(S.shape[0]):
            if F[t, y, x]:
                last = t
            if S[t, y, x]:
                assert (t - last) in allowed


def test_sleep_rewrite_is_bounded_and_resets_sigma4():
    """睡眠の書き込みは、係数の1回の動きが ±1 以内（いつもの書き方）。σ₄ は1歩先の σ と新しい係数から置き直し、下限より下がらない"""
    cell = C.Cell3(sleep=True)
    recs, states = run(cell, 400, P=12.0)
    assert any(bool(r["sleep"].any()) for r in recs)          # 実際に寝ている
    for t in range(1, len(states)):
        m = recs[t]["sleep"]
        if m.any():
            dw = (states[t].core.w - states[t - 1].core.w).abs()
            assert bool((dw[m.expand_as(dw)] <= 1.0 + 1e-6).all())
            v1 = states[t].core.v.clamp(min=Physio2.sigma_min ** 2)
            exp = (v1 * C.gain4(states[t].core.w) / C.gain1(states[t].core.w)).clamp(min=Physio2.sigma_min ** 2)
            assert torch.allclose(states[t].core.vk[m], exp[m])
    assert bool((torch.stack([s.core.vk for s in states]) >= Physio2.sigma_min ** 2 - 1e-12).all())


def test_deterministic_no_dice_at_runtime():
    """実行時の判断にサイコロを使わない（原則3）：同じ世界の乱数なら、2回回して1ビットも同じ"""
    _, s1 = run(C.Cell3(sleep=True), 500, P=12.0, P2=20.0, t_change=250, seed=5)
    _, s2 = run(C.Cell3(sleep=True), 500, P=12.0, P2=20.0, t_change=250, seed=5)
    for a, b in zip(s1[-1].core, s2[-1].core):
        assert torch.equal(a, b)
    assert torch.equal(s1[-1].RL, s2[-1].RL) and torch.equal(s1[-1].q, s2[-1].q) and torch.equal(s1[-1].k, s2[-1].k)


def test_memory_numbers():
    """メモリの会計（1セルあたりの状態の数）：床25、眠る段36（DESIGN.md 10.4）"""
    assert C.memory_numbers(sleep=False) == 25
    assert C.memory_numbers(sleep=True) == 36


def test_dev_code_refuses_sealed_seeds():
    """開発の評価は封印の乱数（9000以上）を拒む"""
    import evaluate as E
    try:
        E.run(C.Cell3(sleep=True), W.WorldConfig(size=H, zone=H), 9301, "grow", plan=E.Plan(n_per=1, dwell=20, window=5, end=(3, 5)))
    except ValueError:
        return
    raise AssertionError("封印の乱数が拒まれなかった")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} 件すべて通過")
