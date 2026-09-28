"""
Step 2 の世界（DESIGN.md 1章）。

32×32 の格子。左端（x=0）の光源が周期 P で振動し、その値が右へ v 列/ステップで流れる（v=0.25 なら4ステップに1列）。
右下 8×8（x ≥ 24 かつ y ≥ 24）はノイズの区域で、毎ステップ一様乱数 U[0,1] を出す（波は届かない）。

    s(x, t) = 0.5 + 0.4·sin(φ(t − x/v)) ＋ N(0, noise²)

- 光源の位相 φ は毎ステップ 2π/P ずつ進む。周期を変えるときは P だけを変え、φ はつなげたままにする
- 列 x は光源の x/v ステップ前の位相を見る。そのため位相の履歴を持つ（長さ (W−1)/v ＋ 1）。1/v は整数に限る
- 位相といっしょに「その位相に進んだときの周期」も履歴に持つ。列 x に新しい周期が届いたかは column_period で見る。
  set_period のあとの最初の advance で、新しい周期で進んだ位相が光源に出る。その時刻を t_c とすると、列 x には t_c ＋ x/v に届く
- 乱数は世界専用の torch.Generator から引く（学習の乱数と混ぜない）
- 位相は [0, 2π) に丸めて float64 で持つ（長く回しても精度が落ちない）。センサーの値は float32、形は (B, 1, H, W)

状態（WorldState）はテンソルだけなので、Step 1 と同じようにプールに入れて続きから回せる。

世界の様子を描く（時空図とコマ。runs/world_preview/ に保存）：

    .venv/Scripts/python.exe step2_attention/world.py
"""
import math
from dataclasses import asdict, dataclass

import torch

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class WorldConfig:
    size: int = 32          # 格子の一辺
    v: float = 0.25         # 波の速さ（列/ステップ）。1/v は整数
    amp: float = 0.4        # 波の振幅
    mean: float = 0.5       # 波の平均
    noise: float = 0.02     # センサーのノイズ（標準偏差）。DESIGN.md 2.5 で 0.005 を検討中（未決定）
    zone: int = 24          # ノイズの区域は x ≥ zone かつ y ≥ zone。size 以上なら区域なし

    def __post_init__(self):
        k = round(1 / self.v)
        if k < 1 or abs(k * self.v - 1) > 1e-9:
            raise ValueError(f"1/v は整数に限る（v={self.v}）")

    @property
    def lag(self) -> int:
        """1列進むのにかかるステップ数（1/v）"""
        return round(1 / self.v)

    @property
    def hist_len(self) -> int:
        return (self.size - 1) * self.lag + 1

    def to_dict(self):
        return asdict(self)


@dataclass
class WorldState:
    phase: torch.Tensor     # (B, L) float64。phase[:, k] は光源の k ステップ前の位相 φ(t−k)
    period: torch.Tensor    # (B, L) float64。period[:, k] は φ(t−k) に進んだときの周期
    P: torch.Tensor         # (B,) float64。次に進むときの周期

    def index(self, idx):
        return WorldState(self.phase[idx], self.period[idx], self.P[idx])

    def to(self, device):
        return WorldState(self.phase.to(device), self.period.to(device), self.P.to(device))


def _periods(P, n, device):
    P = torch.as_tensor(P, dtype=torch.float64, device=device)
    return P.expand(n).clone() if P.dim() == 0 else P.clone()


def reset(cfg: WorldConfig, P, n=1, phase0=None, generator=None, device="cpu") -> WorldState:
    """光源がずっと周期 P で振動していたとして、位相の履歴を埋める（前線はない状態から始まる）。
    P は数か (B,) のテンソル。phase0（光源の今の位相）を省くと、generator から一様に引く"""
    P = _periods(P, n, device)
    B = P.shape[0]
    if phase0 is None:
        phase0 = torch.rand(B, generator=generator, device=device, dtype=torch.float64) * TWO_PI
    else:
        phase0 = torch.as_tensor(phase0, dtype=torch.float64, device=device).expand(B).clone()
    k = torch.arange(cfg.hist_len, device=device, dtype=torch.float64)
    phase = torch.remainder(phase0[:, None] - k[None, :] * TWO_PI / P[:, None], TWO_PI)
    period = P[:, None].expand(B, cfg.hist_len).clone()
    return WorldState(phase, period, P)


def set_period(state: WorldState, P_new, mask=None) -> WorldState:
    """次の一歩から周期を P_new にする（位相はつなげたまま）。mask（(B,) の bool）があれば、その世界だけ"""
    P_new = _periods(P_new, state.P.shape[0], state.P.device)
    P = P_new if mask is None else torch.where(mask, P_new, state.P)
    return WorldState(state.phase, state.period, P)


def advance(state: WorldState) -> WorldState:
    """光源を1ステップ進め、履歴を1つずらす"""
    new = torch.remainder(state.phase[:, :1] + TWO_PI / state.P[:, None], TWO_PI)
    phase = torch.cat([new, state.phase[:, :-1]], 1)
    period = torch.cat([state.P[:, None], state.period[:, :-1]], 1)
    return WorldState(phase, period, state.P)


def column_phase(state: WorldState, cfg: WorldConfig) -> torch.Tensor:
    """(B, W)：列 x が見ている位相 φ(t − x/v)"""
    return state.phase[:, ::cfg.lag]


def column_period(state: WorldState, cfg: WorldConfig) -> torch.Tensor:
    """(B, W)：列 x が見ている位相に進んだときの周期（新しい周期が届いたかの目印）"""
    return state.period[:, ::cfg.lag]


def zone_mask(cfg: WorldConfig, device="cpu") -> torch.Tensor:
    """(H, W) の bool：ノイズの区域。行が y、列が x"""
    y = torch.arange(cfg.size, device=device)[:, None]
    x = torch.arange(cfg.size, device=device)[None, :]
    return (x >= cfg.zone) & (y >= cfg.zone)


def clean_wave(state: WorldState, cfg: WorldConfig) -> torch.Tensor:
    """(B, 1, H, W) float32：ノイズを入れない波の値。ノイズの区域も波の値のまま（テスト・描画用）"""
    s_col = cfg.mean + cfg.amp * torch.sin(column_phase(state, cfg))
    return s_col.float()[:, None, None, :].expand(-1, 1, cfg.size, cfg.size)


def sense(state: WorldState, cfg: WorldConfig, generator=None) -> torch.Tensor:
    """(B, 1, H, W) float32：センサーの値。波＋ノイズ N(0, noise²)、ノイズの区域は一様乱数 U[0,1]"""
    wave = clean_wave(state, cfg)
    shape, dev = wave.shape, wave.device
    s = wave + cfg.noise * torch.randn(shape, generator=generator, device=dev) if cfg.noise > 0 else wave.clone()
    zm = zone_mask(cfg, dev)
    if bool(zm.any()):
        s = torch.where(zm, torch.rand(shape, generator=generator, device=dev), s)
    return s


def sample_periods(n, generator=None, lo=10.0, hi=30.0, hole=(18.0, 22.0), device="cpu") -> torch.Tensor:
    """学習で見せる周期：[lo, hi] から一様に。ただし hole（学習で見せない周期、内側の穴）は除く"""
    a, b = hole
    left, right = a - lo, hi - b
    u = torch.rand(n, generator=generator, device=device, dtype=torch.float64) * (left + right)
    return torch.where(u < left, lo + u, b + (u - left))


def preview(out_dir, cfg=WorldConfig(), P1=12.0, P2=24.0, t_change=200, T=400, seed=0):
    """世界の様子を描く：1行目（波だけの行）の時空図に、前線の理屈の線（t_c ＋ x/v）を重ねたものと、いくつかのコマ"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE, _font  # noqa: E402

    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    st = reset(cfg, P1, n=1, phase0=0.0)
    frames, rows = [], []
    t_c = None
    for t in range(T):
        if t == t_change:
            st = set_period(st, P2)
        s = sense(st, cfg, g)[0, 0].numpy()
        frames.append(s)
        rows.append(s[0])
        st = advance(st)
        if t == t_change:
            t_c = t + 1          # 新しい周期で進んだ最初の位相が光源に出る時刻
    st_img = np.stack(rows)      # (T, W)

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(5.6, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(st_img, aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              extent=(-0.5, cfg.size - 0.5, T - 0.5, -0.5))
    xs = np.arange(cfg.size)
    ax.plot(xs, t_c + xs * cfg.lag, color=SERIES[1], linewidth=2, label=f"前線の理屈の線 t = t_c ＋ x/v（t_c={t_c}）")
    ax.set_xlabel("列 x（光源は x=0）", color=INK_2, fontsize=9)
    ax.set_ylabel("ステップ t", color=INK_2, fontsize=9)
    ax.set_title(f"時空図（y=0 の行）：周期 {P1:g} → {P2:g}、速さ {cfg.v} 列/ステップ、ノイズ {cfg.noise}",
                 loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "spacetime.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # コマ：変える前・前線が真ん中・前線が右端に届いたころ
    picks = [t_change - 1, t_c + 16 * cfg.lag, t_c + (cfg.size - 1) * cfg.lag, T - 1]
    scale, pad, lab = 6, 8, 20
    cell = cfg.size * scale
    font = _font(13)
    img = Image.new("RGB", (pad + len(picks) * (cell + pad), lab + cell + 2 * pad), SURFACE)
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for i, t in enumerate(picks):
        arr = (np.clip(frames[t], 0, 1) * 255).astype(np.uint8)
        tile = Image.fromarray(arr).resize((cell, cell), Image.NEAREST).convert("RGB")
        x0 = pad + i * (cell + pad)
        d.text((x0, 4), f"t={t}", fill=INK_2, font=font)
        img.paste(tile, (x0, lab + pad))
    img.save(out_dir / "frames.png")
    gif = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST)
           for f in frames[::2]]
    gif[0].save(out_dir / "world.gif", save_all=True, append_images=gif[1:], duration=40, loop=0)
    return out_dir


if __name__ == "__main__":
    from pathlib import Path
    out = preview(Path(__file__).resolve().parent / "runs" / "world_preview")
    print("出力:", out)
