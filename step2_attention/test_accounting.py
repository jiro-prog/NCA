"""
Step 2 のセルの約束を確かめるテスト（DESIGN.md 9章）。世界のテストは test_world.py。

    .venv/Scripts/python.exe step2_attention/test_accounting.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, HIDDEN_CH, N_HIST, Physio2, State2, init_state, ls_solution  # noqa: E402
from sealed import check_dev_seed  # noqa: E402

H = Wd = 12


def busy_dna(seed=0):
    """最終層をゼロでなくして、入力によって出力が変わる DNA"""
    torch.manual_seed(seed)
    m = AttentionCell(dna=True)
    torch.nn.init.normal_(m.rule[-1].weight, std=0.3)
    return m


def run_world(m, T, P=12.0, noise=0.0, dense=True, n=1, seed=0):
    cfg = W.WorldConfig(size=H, zone=H, noise=noise)             # 区域なし（テストは波だけ）
    g = torch.Generator().manual_seed(seed)
    ws = W.reset(cfg, P, n=n, generator=g)
    st = init_state(n, H, Wd)
    recs = []
    with torch.no_grad():
        for _ in range(T):
            st, rec = m.step(st, W.sense(ws, cfg, g), dense=dense)
            recs.append(rec)
            ws = W.advance(ws)
    return st, recs


def test_non_writing_cells_keep_their_coefficients():
    """発火の会計：書き換えないセル（2b で引き金を引かなかったセル）は、予測器の係数も隠れの状態も1ビットも変えない"""
    m = busy_dna()
    st = init_state(1, H, Wd)
    st = st._replace(w=torch.randn(1, 3, H, Wd), h=torch.randn(1, HIDDEN_CH, H, Wd),
                     v=torch.full((1, 1, H, Wd), 1.0))                       # σ が大きいので z² は小さく、引き金は引かれない
    with torch.no_grad():
        new, rec = m.step(st, torch.rand(1, 1, H, Wd), dense=False)
    quiet = ~rec["write"]
    assert quiet.all()
    assert torch.equal(new.w, st.w) and torch.equal(new.h, st.h)


def test_only_triggered_cells_write_in_2b():
    """2b：引き金を引いたセルだけが書き換える。2a（dense）なら全員が書き換える"""
    m = busy_dna()
    st = init_state(1, H, Wd)
    c = torch.zeros(1, 1, H, Wd)
    c[0, 0, 3, 3] = 10.0                                               # このセルだけ濃度がしきい値を超えている
    st = st._replace(c=c, v=torch.full((1, 1, H, Wd), 1.0), w=torch.randn(1, 3, H, Wd))
    with torch.no_grad():
        new, rec = m.step(st, torch.full((1, 1, H, Wd), 0.5), dense=False)
    assert int(rec["write"].sum()) == 1 and bool(rec["write"][0, 0, 3, 3])
    changed = (new.w != st.w).any(1)[0]
    assert changed.sum() <= 1
    with torch.no_grad():
        _, rec2 = m.step(st, torch.full((1, 1, H, Wd), 0.5), dense=True)
    assert rec2["write"].all()
    assert int(rec2["trig"].sum()) == 1                                # 2a でも引き金は同じように数える


def test_sigma_never_below_floor():
    """σ は下限を下回らない（誤差が0の世界でも）"""
    m = AttentionCell(dna=False)
    _, recs = run_world(m, 200, noise=0.0)
    assert min(float(r["sigma"].min()) for r in recs) >= Physio2.sigma_min - 1e-9


def test_write_is_bounded():
    """1回の書き換えで各係数が動くのは、減衰ぶんを除いて ±1 以内（tanh）"""
    m = busy_dna()
    torch.nn.init.normal_(m.rule[-1].weight, std=5.0)                   # 大きな出力でも
    st = init_state(1, H, Wd)._replace(w=torch.randn(1, 3, H, Wd) * 3)
    with torch.no_grad():
        new, rec = m.step(st, torch.rand(1, 1, H, Wd), dense=True)
    g = rec["g"]
    assert float((new.w - g * st.w).abs().max()) <= 1.0 + 1e-6


def test_hidden_state_is_bounded():
    """隠れの状態は減衰つきなので、毎ステップ書き換えても |h| ≤ 1/(1−0.9) = 10 に収まる"""
    m = busy_dna()
    torch.nn.init.normal_(m.rule[-1].weight, std=5.0)
    _, _ = run_world(m, 300, noise=0.01)
    st = init_state(1, H, Wd)
    cfg = W.WorldConfig(size=H, zone=H, noise=0.01)
    g = torch.Generator().manual_seed(1)
    ws = W.reset(cfg, 12.0, n=1, generator=g)
    with torch.no_grad():
        for _ in range(300):
            st, _ = m.step(st, W.sense(ws, cfg, g), dense=True)
            ws = W.advance(ws)
    assert float(st.h.abs().max()) <= 1 / (1 - Physio2.h_decay) + 1e-4


def test_least_squares_reaches_exact_coefficients():
    """前処理つきのデルタ則：ノイズなしの正弦波なら、DNA なし（毎ステップ最小二乗の解へ）で数十ステップのうちに正確な係数になる"""
    P = 12.0
    m = AttentionCell(dna=False)
    st, recs = run_world(m, 60, P=P, noise=0.0)
    c = math.cos(2 * math.pi / P)
    exact = torch.tensor([2 * c, -1.0, 1 - c])
    assert torch.allclose(st.w[0, :, 5, 5], exact, atol=2e-2), st.w[0, :, 5, 5]
    assert float(recs[-1]["e"].abs().max()) < 1e-2


def test_ls_solution_matches_numpy():
    """最小二乗の解の計算（対称行列の組み立て）が正しい"""
    torch.manual_seed(0)
    F_ = torch.randn(20, 3)
    y = torch.randn(20)
    R = F_.T @ F_
    r = F_.T @ y
    R6 = torch.tensor([R[0, 0], R[0, 1], R[0, 2], R[1, 1], R[1, 2], R[2, 2]]).view(1, 6, 1, 1)
    sol = ls_solution(R6, r.view(1, 3, 1, 1), 0.0)[0, :, 0, 0]
    ref = torch.linalg.lstsq(F_, y.unsqueeze(1)).solution[:, 0]
    assert torch.allclose(sol, ref, atol=1e-4)


def test_perception_sees_left_neighbor_raw_and_not_R():
    """知覚は3×3の生の値：左隣の係数が、そのまま DNA の入力に入っている。近所の R・r は入らない（発火の会計）"""
    m = AttentionCell()
    st = init_state(1, H, Wd)
    w = torch.zeros(1, 3, H, Wd)
    w[0, :, 5, 4] = torch.tensor([1.25, -0.5, 0.75])                     # (y=5, x=4) の係数
    st = st._replace(w=w, R=torch.full((1, 6, H, Wd), 123.0), r=torch.full((1, 3, H, Wd), 456.0))
    z = m.perceive(m.neighbor_channels(st), torch.zeros(1, 3, H, Wd))
    C = AttentionCell.N_NBR
    left = 3                                                           # unfold の並び：3×3 の (1,0) が左隣
    for k in range(3):
        assert float(z[0, k * 9 + left, 5, 5]) == float(w[0, k, 5, 4])  # (5,5) のセルから見た左隣の係数
    assert z.shape[1] == C * 9 + 3
    assert not ((z == 123.0) | (z == 456.0)).any()


def test_dw_can_copy_left_neighbor():
    """補正 Δw で、係数を左隣の値に近づけられる（書き込みの道がある）"""
    target = torch.tensor([1.5, -0.8, 0.2])
    st = init_state(1, H, Wd)
    w = torch.zeros(1, 3, H, Wd)
    w[0, :, :, 4] = target.view(3, 1)
    st = st._replace(w=w)
    m = AttentionCell()
    with torch.no_grad():
        m.rule[-1].bias[0] = -30.0                                     # η ≈ 0（最小二乗の解は使わない）
        m.rule[-1].bias[1] = 30.0                                      # g ≈ 1
    left = 3
    idx = [k * 9 + left for k in range(3)]
    # 第1層で左隣の係数だけを取り出し、第2層で Δw = atanh(0.9)×(左隣の係数 − 自分の係数) の形を作るかわりに、
    # 出力を直接つなぐ：Δw_k = 左隣の w_k（自分は0なので差と同じ）
    with torch.no_grad():
        m.rule[0].weight.zero_()
        m.rule[0].bias.zero_()
        for k in range(3):
            m.rule[0].weight[k, idx[k], 0, 0] = 1.0                  # 正の部分
            m.rule[0].weight[3 + k, idx[k], 0, 0] = -1.0             # 負の部分
        m.rule[-1].weight.zero_()
        for k in range(3):
            m.rule[-1].weight[2 + k, k, 0, 0] = 1.0
            m.rule[-1].weight[2 + k, 3 + k, 0, 0] = -1.0
    with torch.no_grad():
        new, _ = m.step(st, torch.full((1, 1, H, Wd), 0.5), dense=True)
    got = new.w[0, :, 5, 5]
    assert torch.allclose(got, torch.tanh(target), atol=1e-3)           # 1回で tanh(左隣の値) まで近づく（±1 以内の上限）
    assert torch.sign(got).eq(torch.sign(target)).all()


def test_dna_starts_like_dna_less():
    """DNA のはじめ（最終層ゼロ）は、DNA なし（η=1、g=1、Δw=0）とほぼ同じ書き換え"""
    a, b = AttentionCell(dna=True), AttentionCell(dna=False)
    sa, _ = run_world(a, 80, noise=0.005)
    sb, _ = run_world(b, 80, noise=0.005)
    assert float((sa.w - sb.w).abs().max()) < 0.05


def test_sealed_seeds_refused():
    """封印テストのシード（9000以上）は、開発では使えない"""
    for s in (0, 99, 8999):
        check_dev_seed(s)
    for s in (9000, 9001, 9104):
        try:
            check_dev_seed(s)
        except ValueError:
            continue
        raise AssertionError(f"封印のシード {s} が通ってしまった")


def test_gradient_flows_through_rollout():
    """DNA の勾配は、ロールアウトを通して流れる（係数の再帰を通して）"""
    m = busy_dna()
    cfg = W.WorldConfig(size=H, zone=H, noise=0.005)
    g = torch.Generator().manual_seed(0)
    ws = W.reset(cfg, 12.0, n=2, generator=g)
    st = init_state(2, H, Wd)
    loss = 0
    for _ in range(12):
        st, rec = m.step(st, W.sense(ws, cfg, g), dense=True)
        loss = loss + (rec["e"] ** 2 / (2 * rec["sigma"].detach() ** 2)).mean()
        ws = W.advance(ws)
    loss.backward()
    for name, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def quiet_state():
    """学び終えたように静かな状態（σ が大きく、驚かない）"""
    st = init_state(1, H, Wd)
    return st._replace(v=torch.full((1, 1, H, Wd), 1.0), w=torch.zeros(1, 3, H, Wd))


def test_alarm_keeps_firing_only_when_set():
    """目覚まし：驚かなくても、d が大きいと濃度が溜まって続けて発火する。d が0 なら発火しない"""
    m = AttentionCell(dna=True, alarm=True)
    sense = torch.full((1, 1, H, Wd), 0.0)                      # μ=0 と一致するので驚かない
    for d0, expect in [(Physio2.theta, True), (0.0, False)]:
        st = quiet_state()._replace(d=torch.full((1, 1, H, Wd), d0))
        fired = 0
        with torch.no_grad():
            for _ in range(12):
                st, rec = m.step(st, sense, policy={"kind": "alarm"})
                fired += int(rec["hard"].sum())
        assert (fired > 0) == expect, (d0, fired)


def test_alarm_ste_reaches_alarm_output():
    """目覚ましのときだけ、引き金の STE を通して、目覚ましの出力（最終層の最後の行）に勾配が届く"""
    torch.manual_seed(0)
    m = AttentionCell(dna=True, alarm=True)
    torch.nn.init.normal_(m.rule[-1].weight, std=0.1)
    cfg = W.WorldConfig(size=H, zone=H, noise=0.005)
    g = torch.Generator().manual_seed(0)
    ws = W.reset(cfg, 12.0, n=2, generator=g)
    st = init_state(2, H, Wd)
    loss = 0
    for _ in range(15):
        st, rec = m.step(st, W.sense(ws, cfg, g), policy={"kind": "alarm"})
        loss = loss + rec["fire"].mean() + (rec["e"] ** 2 / (2 * rec["sigma"].detach() ** 2)).mean()
        ws = W.advance(ws)
    loss.backward()
    ga = m.rule[-1].weight.grad[-1]
    assert torch.isfinite(ga).all() and ga.abs().sum() > 0


def test_non_alarm_policies_have_no_ste():
    """目覚まし以外では、引き金は学習しない：発火（学習用）は 0/1 そのもので、濃度へ勾配を流さない"""
    m = busy_dna()
    st = init_state(1, H, Wd)
    for pol in ({"kind": "trigger"}, {"kind": "cand3", "K": 20}, {"kind": "dense"}):
        with torch.no_grad():
            _, rec = m.step(st, torch.rand(1, 1, H, Wd), policy=pol)
        assert torch.equal(rec["fire"], rec["write"].float()), pol


def test_cand3_window_rules():
    """候補3：引き金で窓が開き、驚いている間は毎ステップ発火、窓の中では濃度を0に戻さない。驚きが収まるか K 回で閉じる"""
    m = AttentionCell(dna=False)
    st = quiet_state()
    c = torch.zeros(1, 1, H, Wd)
    c[0, 0, 5, 5] = 100.0                                          # (5,5) だけ大きく驚いている
    st = st._replace(c=c)
    sense = torch.zeros(1, 1, H, Wd)
    fires = []
    with torch.no_grad():
        for _ in range(8):
            st, rec = m.step(st, sense, policy={"kind": "cand3", "K": 5})
            fires.append(bool(rec["hard"][0, 0, 5, 5]))
    assert fires[:5] == [True] * 5 and fires[5] is False            # 上限5回で閉じ、1ステップ休む
    st = quiet_state()._replace(c=torch.full((1, 1, H, Wd), 4.5))    # しきい値のすぐ上：減衰でしきい値を下回ったら閉じる
    with torch.no_grad():
        st, rec = m.step(st, sense, policy={"kind": "cand3", "K": 20})
        n_open = int(rec["hard"].sum())
        st, rec = m.step(st, sense, policy={"kind": "cand3", "K": 20})
    assert n_open == H * Wd and int(rec["hard"].sum()) == H * Wd and int(rec["write"].sum()) == 0   # 2回目は驚きが収まって、書かずに閉じる


def test_2a_weights_still_load():
    """目覚ましの出力を足しても、2a の重み（出力9つ）はそのまま読める"""
    p = Path(__file__).resolve().parent / "runs" / "2a_seed0" / "model.pt"
    if not p.exists():
        return
    m = AttentionCell(dna=True)
    m.load_state_dict(torch.load(p, map_location="cpu"))
    assert m.rule[-1].weight.shape[0] == AttentionCell.N_OUT


def test_one_step_surprise_is_unchanged():
    """驚きの信号が1歩先（k=1、これまでどおり）なら、覚えている予測と σ_k は1歩先の予測と σ にぴったり重なり、濃度に溜めるのも z²"""
    assert Physio2.surprise_ahead == 1
    cfg = W.WorldConfig(size=H, zone=H, noise=0.01)
    g = torch.Generator().manual_seed(3)
    ws = W.reset(cfg, 12.0, n=2, generator=g)
    m = busy_dna()
    st = init_state(2, H, Wd)
    assert st.buf.shape[1] == 1
    with torch.no_grad():
        for _ in range(30):
            st, rec = m.step(st, W.sense(ws, cfg, g), dense=False)
            assert torch.equal(rec["zk2"], rec["z2"])
            assert torch.equal(st.buf[:, 0:1], st.mu) and torch.equal(st.vk, st.v)
            ws = W.advance(ws)


def ahead4(fn):
    """驚きの信号を4歩先にして回し、終わったら元に戻す"""
    def wrapped():
        old = Physio2.surprise_ahead
        Physio2.surprise_ahead = 4
        try:
            fn()
        finally:
            Physio2.surprise_ahead = old
    wrapped.__name__ = fn.__name__
    return wrapped


@ahead4
def test_four_step_non_firing_cells_keep_their_coefficients():
    """4歩先の驚きでも、発火の会計は同じ：引き金を引かなかったセルは、係数も隠れの状態も1ビットも変えない"""
    m = busy_dna()
    st = init_state(1, H, Wd)
    assert st.buf.shape[1] == 4
    st = st._replace(w=torch.randn(1, 3, H, Wd), h=torch.randn(1, HIDDEN_CH, H, Wd),
                     vk=torch.full((1, 1, H, Wd), 1.0))                      # σ₄ が大きいので z₄² は小さく、引き金は引かれない
    with torch.no_grad():
        new, rec = m.step(st, torch.rand(1, 1, H, Wd), dense=False)
    assert (~rec["write"]).all()
    assert torch.equal(new.w, st.w) and torch.equal(new.h, st.h)


@ahead4
def test_four_step_cheap_updates_every_step():
    """4歩先の予測と σ₄ は、発火しないセルも毎ステップ更新する安い式：覚えている予測は1つずれて、いちばん新しいものは係数を4回まわした予測"""
    m = AttentionCell(dna=False)
    cfg = W.WorldConfig(size=H, zone=H, noise=0.01)
    g = torch.Generator().manual_seed(4)
    ws = W.reset(cfg, 12.0, n=1, generator=g)
    st = init_state(1, H, Wd)._replace(v=torch.full((1, 1, H, Wd), 1.0), vk=torch.full((1, 1, H, Wd), 1.0))
    with torch.no_grad():
        for _ in range(6):
            sense = W.sense(ws, cfg, g)
            new, rec = m.step(st, sense, dense=False)
            assert torch.equal(new.buf[:, :3], st.buf[:, 1:])
            w, x0, x1 = new.w, sense, st.hist[:, 0:1]
            p = x0
            q = x1
            for _ in range(4):
                p, q = w[:, 0:1] * p + w[:, 1:2] * q + w[:, 2:3], p
            assert torch.allclose(new.buf[:, 3:4], p)
            assert not torch.equal(new.vk, st.vk)                           # σ₄ は全員が毎ステップ動く
            assert torch.allclose(rec["zk2"], rec["ek"] ** 2 / torch.clamp(st.vk, min=Physio2.sigma_min ** 2), rtol=1e-5)
            st = new
            ws = W.advance(ws)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} 件すべて通過")
