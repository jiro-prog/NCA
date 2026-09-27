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


def busy_model():
    """最終層をゼロでなくして、発火すれば必ず状態が変わるモデル"""
    torch.manual_seed(0)
    m = ConcentrationNCA()
    torch.nn.init.normal_(m.rule[-1].weight, std=0.05)
    return m


def all_alive_state(c=0.0, d=0.0):
    """全セルが生きていて（α=1）、誕生済みで、静かな状態"""
    x = torch.rand(1, CHANNELS, H, W)
    x[:, 3] = 1.0
    z = torch.zeros(1, 1, H, W)
    return State(x, z + c, z + d, z.clone(), z.clone(), torch.ones_like(z, dtype=torch.bool))


def test_non_firing_cells_do_nothing():
    """発火していない生きたセルは、状態も目覚ましも1ビットも変わらない"""
    m = busy_model()
    s = all_alive_state()
    g = torch.Generator().manual_seed(1)
    c = torch.rand(1, 1, H, W, generator=g) * 1.5                     # 半分くらいがしきい値を超える
    r = (torch.rand(1, 1, H, W, generator=g) < 0.3).float()           # 3割は不応期
    s = State(s.x, c, s.d, r, s.e, s.was)
    with torch.no_grad():
        new, _, hard, pre = m.step(s)
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
        s, _, hard1, _ = m.step(s)
        s, _, hard2, _ = m.step(s)
    assert (hard1 & hard2).sum() == 0


def test_single_neighbor_event_is_subthreshold():
    """近所1つの変化イベント（上限の大きさ）だけでは発火しない。3つそろうと発火する"""
    m = busy_model()
    s = all_alive_state()
    e = torch.zeros(1, 1, H, W)
    e[0, 0, 10, 10] = Physio.e_max
    with torch.no_grad():
        _, _, hard, _ = m.step(State(s.x, s.c, s.d, s.r, e, s.was))
    assert hard.sum() == 0
    e[0, 0, 10, 12] = Physio.e_max
    e[0, 0, 12, 11] = Physio.e_max
    with torch.no_grad():
        _, _, hard, _ = m.step(State(s.x, s.c, s.d, s.r, e, s.was))
    assert hard[0, 0, 11, 11]                          # 3つのイベントの共通の近所は 3×0.4 ≥ しきい値


def test_seed_fires_by_birth():
    """種とその近所（3×3）は、最初のステップで誕生して発火する。それ以外は発火しない"""
    m = ConcentrationNCA()
    s = seed_state(H, W)
    with torch.no_grad():
        _, _, hard, pre = m.step(s)
    assert torch.equal(hard, pre) and hard.sum() == 9


def test_gradient_flows():
    """ロスと発火のコストから、ルールのネットの全パラメータに有限の勾配が流れる"""
    m = busy_model()
    s = seed_state(H, W, n=2)
    out, fire_sum, _, _ = m(s, 12)
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
        out, fire_sum, _, _ = m(seed_state(H, W, n=2), 8)
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
