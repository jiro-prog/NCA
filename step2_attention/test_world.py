"""
Step 2 の世界（world.py）の約束を確かめるテスト（DESIGN.md 1章・9章）。

    .venv/Scripts/python.exe step2_attention/test_world.py
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from world import (TWO_PI, WorldConfig, advance, clean_wave, column_period, reset, sample_periods,  # noqa: E402
                   sense, set_period, zone_mask)

CFG = WorldConfig()
QUIET = replace(CFG, noise=0.0)


def run(cfg, P1, T, change_at=None, P2=None, n=2, seed=0, noisy=True):
    """T ステップ回して、センサーの値 (T, B, H, W) と列ごとの周期 (T, B, W) を返す"""
    g = torch.Generator().manual_seed(seed)
    st = reset(cfg, P1, n=n, generator=g)
    S, CP = [], []
    for t in range(T):
        if change_at is not None and t == change_at:
            st = set_period(st, P2)
        S.append((sense(st, cfg, g) if noisy else clean_wave(st, cfg))[:, 0])
        CP.append(column_period(st, cfg))
        st = advance(st)
    return torch.stack(S), torch.stack(CP)


def test_phase_continuous_on_period_change():
    """周期を変えても位相は飛ばない：光源の位相は毎ステップ、その時の周期のぶんだけ進む"""
    st = reset(QUIET, 12.0, n=1, phase0=1.0)
    src = [st.phase[0, 0].item()]
    for t in range(60):
        if t == 30:
            st = set_period(st, 24.0)
        st = advance(st)
        src.append(st.phase[0, 0].item())
    steps = [(b - a) % TWO_PI for a, b in zip(src[:-1], src[1:])]
    for t, d in enumerate(steps):
        want = TWO_PI / (12.0 if t < 30 else 24.0)
        assert abs(d - want) < 1e-9, (t, d, want)
    # 光源の値もなめらか（1ステップの変化は 振幅×2π/P を超えない）
    S, _ = run(QUIET, 12.0, 60, change_at=30, P2=24.0, n=1, noisy=False)
    jump = (S[1:, 0, 0, 0] - S[:-1, 0, 0, 0]).abs().max().item()
    assert jump <= QUIET.amp * TWO_PI / 12.0 + 1e-6, jump


def test_new_period_arrives_after_x_over_v():
    """列 x に新しい周期が届くのは、光源に出てから x/v ステップ後（v=0.25 なら 4x ステップ後）"""
    t_set = 40
    _, CP = run(QUIET, 12.0, t_set + 1 + 31 * 4 + 10, change_at=t_set, P2=24.0, n=1, noisy=False)
    t_c = t_set + 1     # set_period のあとの最初の advance で、新しい周期の位相が光源に出る
    for x in range(QUIET.size):
        col = CP[:, 0, x]
        first = int(torch.nonzero(col == 24.0)[0])
        assert first == t_c + x * QUIET.lag, (x, first)
        assert (col[first:] == 24.0).all() and (col[:first] == 12.0).all()


def test_noiseless_wave_is_exact_ar2():
    """ノイズなしの波は、2つ前までの線形予測で正確に当たる：s(t+1) = 2cos(ω)s(t) − s(t−1) ＋ 2(1−cos ω)·平均"""
    for P in (10.0, 12.0, 20.0, 36.0):
        S, _ = run(QUIET, P, 80, noisy=False)
        c = math.cos(TWO_PI / P)
        pred = 2 * c * S[1:-1] - S[:-2] + 2 * (1 - c) * QUIET.mean
        assert (S[2:] - pred).abs().max().item() < 1e-5, P


def test_wave_travels_at_v():
    """波は右へ v 列/ステップで流れる：左隣の今の値は、自分の 1/v ステップ先の値（DESIGN.md 7.3 ののぞき見の前提）"""
    S, _ = run(QUIET, 12.0, 100, noisy=False)
    k = QUIET.lag
    left_now = S[:-k, :, :, :-1]      # s(x−1, t)
    me_later = S[k:, :, :, 1:]        # s(x, t＋1/v)
    assert torch.allclose(left_now, me_later, atol=1e-6)


def test_plane_wave_rows_equal():
    """波は平面波：同じ列なら、どの行も同じ値（ノイズなし、ノイズの区域を除く）"""
    S, _ = run(QUIET, 12.0, 30, noisy=False)
    zm = zone_mask(QUIET)
    ref = S[:, :, 0:1, :].expand_as(S)
    assert torch.equal(S[..., ~zm], ref[..., ~zm])


def test_noise_zone_is_uniform_and_wave_free():
    """ノイズの区域は一様乱数 U[0,1]：平均0.5、標準偏差 0.289、波と相関しない"""
    S, _ = run(CFG, 12.0, 400, n=8)
    zm = zone_mask(CFG)
    assert zm.sum().item() == 64          # 右下 8×8
    z = S[..., zm]
    assert z.min().item() >= 0.0 and z.max().item() <= 1.0
    assert abs(z.mean().item() - 0.5) < 0.01
    assert abs(z.std().item() - 1 / math.sqrt(12)) < 0.01
    src = S[:, :, 0, 0]                   # 光源の値
    zz = S[:, :, 30, 30]
    corr = torch.corrcoef(torch.stack([src.flatten(), zz.flatten()]))[0, 1].item()
    assert abs(corr) < 0.1, corr


def test_sensor_noise_level():
    """波の区域のノイズは N(0, noise²)：ノイズなしの値との差の標準偏差が noise"""
    for noise in (0.02, 0.005):
        cfg = replace(CFG, noise=noise)
        g = torch.Generator().manual_seed(1)
        st = reset(cfg, 12.0, n=8, generator=g)
        diffs = []
        for _ in range(100):
            diffs.append((sense(st, cfg, g) - clean_wave(st, cfg))[..., ~zone_mask(cfg)])
            st = advance(st)
        sd = torch.cat(diffs).std().item()
        assert abs(sd / noise - 1) < 0.02, (noise, sd)


def test_reproducible_with_seed():
    """同じシードなら同じ世界、違うシードなら違う世界"""
    a, _ = run(CFG, 12.0, 20, seed=3)
    b, _ = run(CFG, 12.0, 20, seed=3)
    c, _ = run(CFG, 12.0, 20, seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_set_period_mask_changes_only_selected_worlds():
    """mask で選んだ世界だけ周期が変わる（プールで一部の世界だけ変えるとき）"""
    st = reset(CFG, torch.tensor([12.0, 12.0, 12.0]), phase0=0.0)
    st = set_period(st, 24.0, mask=torch.tensor([True, False, True]))
    assert st.P.tolist() == [24.0, 12.0, 24.0]


def test_sample_periods_skip_hole():
    """学習で見せる周期は 10〜30 で、18〜22 の穴には入らない。穴の左右からほぼ半分ずつ"""
    g = torch.Generator().manual_seed(0)
    P = sample_periods(20000, g)
    assert P.min().item() >= 10.0 and P.max().item() <= 30.0
    assert not ((P > 18.0) & (P < 22.0)).any()
    assert abs((P < 18.0).float().mean().item() - 0.5) < 0.02


def test_v_must_divide():
    """1/v が整数でない速さは作れない（列ごとの遅れが整数ステップにならないため）"""
    try:
        WorldConfig(v=0.3)
    except ValueError:
        return
    raise AssertionError("v=0.3 が通ってしまった")


def test_v1_control():
    """対照の v=1：新しい周期は1列/ステップで届き、左隣の今の値は自分の1ステップ先"""
    cfg = replace(QUIET, v=1.0)
    assert cfg.lag == 1 and cfg.hist_len == 32
    S, CP = run(cfg, 12.0, 60, change_at=10, P2=24.0, n=1, noisy=False)
    assert torch.allclose(S[:-1, :, :, :-1], S[1:, :, :, 1:], atol=1e-6)
    first = [int(torch.nonzero(CP[:, 0, x] == 24.0)[0]) for x in range(cfg.size)]
    assert first == [11 + x for x in range(cfg.size)]


def test_long_run_keeps_precision():
    """長く回しても位相は [0, 2π) にとどまり、ノイズなしの波は2つ前までの線形予測で当たったまま"""
    st = reset(QUIET, 13.7, n=1, phase0=0.3)
    for _ in range(20000):
        st = advance(st)
    assert (st.phase >= 0).all() and (st.phase < TWO_PI).all()
    S = []
    for _ in range(3):
        S.append(clean_wave(st, QUIET)[0, 0, 0])
        st = advance(st)
    c = math.cos(TWO_PI / 13.7)
    pred = 2 * c * S[1] - S[0] + 2 * (1 - c) * QUIET.mean
    assert (S[2] - pred).abs().max().item() < 1e-5


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} 件すべて通過")
