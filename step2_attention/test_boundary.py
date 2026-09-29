"""
境目の実験（boundary.py）のテスト。いちばん大事なのは、束に並べた1ステップ（bstep）が、本物のセル（model.step、DNA なし）と1ビットも違わないこと。

    .venv/Scripts/python.exe step2_attention/test_boundary.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from boundary import KIND_CLOCK, KIND_DICE, KIND_SURPRISE, WARMUP, bstep, policy_tensors, schedule  # noqa: E402
from model import AttentionCell, Physio2, init_state  # noqa: E402

H = Wd = 12
FIELDS = ("w", "R", "r", "v", "c", "ref", "h", "hist", "mu", "d", "win", "buf", "vk")


def pol(kind, theta=4.0, K=1, k=1, p=0.0, family=""):
    return dict(family=family, name="", kind=kind, theta=theta, K=K, k=k, p=p)


def run_pair(bpol, mpol, T=40, theta=4.0, seed=0):
    """同じ世界を、bstep と model.step で回して、毎ステップの状態と発火を比べる"""
    cfg = W.WorldConfig(size=H, zone=H)
    old_theta = Physio2.theta
    Physio2.theta = theta
    try:
        m = AttentionCell(dna=False)
        g = torch.Generator().manual_seed(seed)
        ws = W.reset(cfg, 12.0, n=2, generator=g)
        sb = init_state(2, H, Wd)
        sm = init_state(2, H, Wd)
        gb = torch.Generator().manual_seed(99)
        gm = torch.Generator().manual_seed(99)
        pt = policy_tensors([bpol], 2, "cpu")
        with torch.no_grad():
            for t in range(T):
                if t == 20:
                    ws = W.set_period(ws, 24.0)
                sense = W.sense(ws, cfg, g)
                sb, rb = bstep(sb, sense, pt, torch.tensor(t), gb)
                mp = dict(mpol)
                if mp["kind"] == "dice":
                    mp["gen"] = gm
                if mp["kind"] == "clock":
                    mp["t"] = t
                sm, rm = m.step(sm, sense, policy=mp)
                assert torch.equal(rb["hard"], rm["hard"]), (bpol, t)
                for f in FIELDS:
                    assert torch.equal(getattr(sb, f), getattr(sm, f)), (bpol, t, f)
                ws = W.advance(ws)
    finally:
        Physio2.theta = old_theta


def test_bstep_matches_model_trigger():
    """驚き駆動 θ=4・K=1 は、本物のセルの元の引き金（trigger）と同じ"""
    run_pair(pol(KIND_SURPRISE, 4.0, 1), {"kind": "trigger"})


def test_bstep_matches_model_cand3():
    """驚き駆動 θ=4・K=5 と θ=2・K=20 は、本物のセルの候補3（しきい値もそろえる）と同じ"""
    run_pair(pol(KIND_SURPRISE, 4.0, 5), {"kind": "cand3", "K": 5})
    run_pair(pol(KIND_SURPRISE, 2.0, 20), {"kind": "cand3", "K": 20}, theta=2.0)


def test_bstep_matches_model_clock_and_dice():
    """時計とサイコロも、本物のセルと同じ（サイコロは同じ乱数の並びで）"""
    run_pair(pol(KIND_CLOCK, k=4), {"kind": "clock", "k": 4})
    run_pair(pol(KIND_DICE, p=0.1), {"kind": "dice", "p": 0.1})


def test_mixed_batch_is_elementwise():
    """束に違う発火のしかたを並べても、それぞれは1つだけで回したときと同じ（サイコロを除く。サイコロは乱数の並びが束の形で変わる）"""
    cfg = W.WorldConfig(size=H, zone=H)
    pols = [pol(KIND_SURPRISE, 3.0, 5), pol(KIND_CLOCK, k=8)]
    g = torch.Generator().manual_seed(1)
    ws = W.reset(cfg, 12.0, n=1, generator=g)
    s_all = init_state(2, H, Wd)
    s_one = [init_state(1, H, Wd) for _ in pols]
    pt_all = policy_tensors(pols, 1, "cpu")
    pt_one = [policy_tensors([p_], 1, "cpu") for p_ in pols]
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for t in range(40):
            if t == 15:
                ws = W.set_period(ws, 20.0)
            sense = W.sense(ws, cfg, g)
            s_all, r_all = bstep(s_all, sense.repeat(2, 1, 1, 1), pt_all, torch.tensor(t), gen)
            for i in range(2):
                s_one[i], r_one = bstep(s_one[i], sense, pt_one[i], torch.tensor(t), gen)
                assert torch.equal(r_all["hard"][i:i + 1], r_one["hard"])
                assert torch.equal(s_all.w[i:i + 1], s_one[i].w)
            ws = W.advance(ws)


def test_warmup_never_writes():
    """はじめの50ステップ（warm=True）は、誰も書き換えない（発火も数えない）"""
    cfg = W.WorldConfig(size=H, zone=H)
    g = torch.Generator().manual_seed(2)
    ws = W.reset(cfg, 12.0, n=1, generator=g)
    pols = [pol(KIND_SURPRISE, 2.0, 20), pol(KIND_CLOCK, k=2), pol(KIND_DICE, p=0.5)]
    s = init_state(3, H, Wd)
    w0 = s.w.clone()
    pt = policy_tensors(pols, 1, "cpu")
    with torch.no_grad():
        for t in range(WARMUP):
            s, r = bstep(s, W.sense(ws, cfg, g).repeat(3, 1, 1, 1), pt, torch.tensor(t - WARMUP), torch.Generator(), warm=True)
            assert not r["hard"].any()
            ws = W.advance(ws)
    assert torch.equal(s.w, w0)


def test_schedule_rules():
    """変化の時刻：最初は書き換えない50ステップのあとの [0, T_c)、CV=0 はちょうど T_c ごと、CV>0 は平均 T_c・ばらつき CV のガンマの間隔"""
    rng = np.random.default_rng(0)
    s0 = schedule(100, 0.0, 5000, rng)
    assert WARMUP <= s0[0] < WARMUP + 100 and all(b - a in (99, 100, 101) for a, b in zip(s0, s0[1:]))
    for cv in (0.5, 1.0, 2.0):
        gaps = []
        for _ in range(200):
            s = schedule(200, cv, 200000, rng)
            gaps += list(np.diff(s))
        gaps = np.array(gaps, dtype=float)
        assert abs(gaps.mean() / 200 - 1) < 0.05, (cv, gaps.mean())
        assert abs(gaps.std() / gaps.mean() - cv) < 0.1 * max(cv, 1), (cv, gaps.std() / gaps.mean())


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} 件すべて通過")
