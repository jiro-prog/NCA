"""
発火の会計（CLAUDE.md の原則7）と、発火の仕組みの約束を確かめるテスト。

    .venv/Scripts/python.exe step1_concentration/test_accounting.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from model import CHANNELS, ConcentrationNCA, Physio, State, seed_state  # noqa: E402

H = W = 24


def busy_model(see_c=False):
    """最終層をゼロでなくして、発火すれば必ず状態が変わるモデル"""
    torch.manual_seed(0)
    m = ConcentrationNCA(see_c=see_c)
    torch.nn.init.normal_(m.rule[-1].weight, std=0.05)
    return m


def all_alive_state(c=0.0, d=0.0):
    """全セルが生きていて（α=1）、誕生済みで、静かな状態"""
    x = torch.rand(1, CHANNELS, H, W)
    x[:, 3] = 1.0
    z = torch.zeros(1, 1, H, W)
    t = torch.ones_like(z, dtype=torch.bool)
    return State(x, z + c, z + d, z.clone(), z.clone(), t, ~t, ~t)


def test_non_firing_cells_do_nothing():
    """発火していない生きたセルは、状態も目覚ましも1ビットも変わらない"""
    for see_c in (False, True):
        m = busy_model(see_c)
        s = all_alive_state()
        g = torch.Generator().manual_seed(1)
        c = torch.rand(1, 1, H, W, generator=g) * 1.5                     # 半分くらいがしきい値を超える
        r = (torch.rand(1, 1, H, W, generator=g) < 0.3).float()           # 3割は不応期
        s = s._replace(c=c, r=r)
        with torch.no_grad():
            new, _, hard, pre, _ = m.step(s)
        live = pre & m.alive(new.x)
        quiet = live & ~hard
        assert hard.any() and quiet.any()
        assert torch.equal(new.x[quiet.expand_as(new.x)], s.x[quiet.expand_as(s.x)])
        assert torch.equal(new.d[quiet], s.d[quiet])
        fired = live & hard
        assert (new.x - s.x).abs().sum(1, keepdim=True)[fired].min() > 0     # 発火したセルは変わる
        assert not (hard & (r > 0)).any()                                    # 不応期のセルは発火しない
        assert torch.all(new.c[hard] == 0) and torch.all(new.r[hard] == Physio.refractory)


def test_refractory_blocks_next_step():
    """発火した次のステップは、濃度がしきい値を超えていても発火しない"""
    m = busy_model()
    s = all_alive_state(c=5.0, d=5.0)
    with torch.no_grad():
        s, _, hard1, _, _ = m.step(s)
        s, _, hard2, _, _ = m.step(s)
    assert (hard1 & hard2).sum() == 0


def test_single_neighbor_event_is_subthreshold():
    """近所1つの変化イベント（上限の大きさ）だけでは発火しない。3つそろうと発火する"""
    m = busy_model()
    s = all_alive_state()
    e = torch.zeros(1, 1, H, W)
    e[0, 0, 10, 10] = Physio.e_max
    with torch.no_grad():
        _, _, hard, _, _ = m.step(s._replace(e=e))
    assert hard.sum() == 0
    e[0, 0, 10, 12] = Physio.e_max
    e[0, 0, 12, 11] = Physio.e_max
    with torch.no_grad():
        _, _, hard, _, _ = m.step(s._replace(e=e))
    assert hard[0, 0, 11, 11]                          # 3つのイベントの共通の近所は 3×0.4 ≥ しきい値


def test_seed_fires_by_birth():
    """種とその近所（3×3）は、最初のステップで誕生して発火する。それが誕生の発火として数えられる"""
    m = ConcentrationNCA()
    s = seed_state(H, W)
    with torch.no_grad():
        _, _, hard, pre, birth = m.step(s)
    assert torch.equal(hard, pre) and hard.sum() == 9
    assert torch.equal(birth, hard)


def test_delayed_fire_happens_exactly_one_step_later():
    """遅らされた発火は消えずに、次のステップで濃度に関係なく必ず起きる。持ち越しをさらに遅らせることはない。
    （2026-09-28 のバグ：遅らされた誕生の発火は、濃度が減衰してしきい値を下回り、永久に消えていた）"""
    m = ConcentrationNCA()
    s = seed_state(H, W)
    with torch.no_grad():
        s1, _, hard1, pre1, _ = m.step(s, jitter_p=1.0)          # 全員の発火を遅らせる
        assert hard1.sum() == 0 and torch.equal(s1.pend, pre1)
        s2, _, hard2, _, birth2 = m.step(s1, jitter_p=1.0)       # 持ち越しは、ずれがあっても起きる
    assert hard2.sum() == 9 and torch.equal(birth2, hard2)         # 生まれて最初の発火なので、誕生として数える
    assert s2.pend.sum() == 0


def test_birth_counted_once_per_life():
    """誕生の発火は、1回の生につき、生まれてから最初の1回だけ（死んで生まれ直したら、また1回）"""
    m = busy_model()
    s = seed_state(H, W)
    births = torch.zeros(1, 1, H, W)
    rebirths = 0
    with torch.no_grad():
        for _ in range(30):
            s, _, hard, pre, birth = m.step(s)
            assert not (birth & ~hard).any()
            births = births * pre.float() + birth.float()      # 生きていないセルは数え直す
            assert births.max() <= 1
            rebirths += int(birth.sum())
    assert rebirths > 9                                        # 種の周り以外でも誕生が起きている


def test_see_c_switch():
    """濃度を入力に入れないモデルの知覚は、元祖と同じ48次元。既定は入れる"""
    assert ConcentrationNCA(see_c=False).kernel.shape[0] == CHANNELS * 3
    assert ConcentrationNCA(see_c=True).kernel.shape[0] == (CHANNELS + 1) * 3
    assert ConcentrationNCA().see_c


def test_coupling_modes():
    """近所8つがそろって上限の変化をしたとき：合計なら発火、平均（0.4）や上限2の合計（0.8）なら発火しない"""
    m = busy_model()
    s = all_alive_state()
    e = torch.zeros(1, 1, H, W)
    e[0, 0, 9:12, 9:12] = Physio.e_max
    e[0, 0, 10, 10] = 0                               # 真ん中のセルの近所8つだけが変化した
    s = s._replace(e=e)
    saved = (Physio.coupling, Physio.coupling_cap)
    try:
        for mode, cap, fires in [("sum", None, True), ("mean", None, False), ("cap", 2.0, False)]:
            Physio.coupling, Physio.coupling_cap = mode, cap
            with torch.no_grad():
                new, _, hard, _, _ = m.step(s)
            assert bool(hard[0, 0, 10, 10]) == fires, mode
            assert abs(float(new.c[0, 0, 10, 10]) - (0.0 if fires else Physio.kappa * {"mean": 1.0, "cap": 2.0}[mode])) < 1e-6
    finally:
        Physio.coupling, Physio.coupling_cap = saved


def test_hardness_one_is_unchanged():
    """固さ1（既定）の動きは、固さを渡さないときとビット単位で同じ"""
    m = busy_model()
    s = all_alive_state(c=0.7, d=0.4)
    with torch.no_grad():
        a = m.step(s)
        b = m.step(s, hardness=1.0)
    for x, y in zip(a[0], b[0]):
        assert torch.equal(x, y)
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_soft_phase_moves_quiet_cells_but_not_the_machinery():
    """固さ0（学習の初期）では、発火していないセルも少し更新される。ただし不応期のセルは動かず、
    濃度のリセットや不応期は 0/1 の発火で決まる"""
    m = busy_model()
    s = all_alive_state(c=0.5)                          # 誰もしきい値に届いていない
    r = torch.zeros(1, 1, H, W)
    r[0, 0, :, :W // 2] = 1                             # 左半分は不応期
    s = s._replace(r=r)
    with torch.no_grad():
        new, _, hard, pre, _ = m.step(s, hardness=0.0)
    assert hard.sum() == 0                              # 0/1 では誰も発火していない
    moved = (new.x - s.x).abs().sum(1, keepdim=True) > 0
    assert moved[0, 0, :, W // 2 + 1:].all()            # 右半分（不応期でない）は少し動く
    assert not moved[0, 0, :, :W // 2].any()            # 不応期のセルは動かない
    assert torch.all(new.r[0, 0, :, W // 2:] == 0)      # 0/1 で発火していないので、不応期にはならない


def test_gradient_flows():
    """ロスと発火のコストから、ルールのネットの全パラメータに有限の勾配が流れる"""
    m = busy_model()
    s = seed_state(H, W, n=2)
    out, fire_sum, _, _, _ = m(s, 12)
    loss = (out.x[:, :4] ** 2).mean() + 0.01 * fire_sum.sum()
    loss.backward()
    for name, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_checkpoint_matches():
    """勾配チェックポイントを使っても、順方向も勾配も同じ"""
    grads = []
    for ck in (False, True):
        m = busy_model()
        m.use_checkpoint = ck
        out, fire_sum, _, _, _ = m(seed_state(H, W, n=2), 8)
        ((out.x[:, :4] ** 2).mean() + 0.01 * fire_sum.sum()).backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} 件すべて通過")
