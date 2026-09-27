"""
成長・維持・傷の3期で測る評価プロトコル。どのモデルも同じ手順で測って比べる。

    0 ── 成長 ──▶ 200 ── 維持 ──▶ 400 [傷] ── 回復 ──▶ 800

- ロス：RGBA と目標の二乗誤差の平均（学習のロスと同じ式）
- 発火：更新前の生死判定で「生きている」セルの中で数える（CLAUDE.md「計算の会計」）

モデルごとの違いは「アダプター」に閉じ込める。アダプターは次を持つ：
    reset()              -> state                      種から始める
    step(state)          -> state, {"fires", "alive"}  1ステップ進める（どちらも (1,1,H,W) の bool）
    rgba(state)          -> (1,4,H,W)                  見た目のチャンネル
    damage(state, hole)  -> state                      hole（(H,W) の bool）のセルを殺す
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class Protocol:
    total_steps: int = 800
    grow_end: int = 200                     # ここまでが成長期
    damage_at: int = 400                    # このステップの直前に傷をつける
    damage_center: tuple = (35.5, 44.5)     # (y, x)。花の中心から右へ9セルの花びら
    damage_radius: float = 8.0              # 学習時の穴と同じ大きさ
    frame_every: int = 4                    # GIFのコマの間隔（発火はこの間の合計で描く）
    thresholds: tuple = (0.02, 0.01, 0.005, 0.002)

    def phase(self, t):
        if t < self.grow_end:
            return "grow"
        return "maintain" if t < self.damage_at else "recover"

    def to_dict(self):
        return asdict(self)


# 抜き出して並べるコマ（コンタクトシート用）
SHEET_STEPS = (0, 25, 50, 100, 199, 399, 400, 404, 420, 450, 600, 799)


def damage_hole(h, w, center, radius, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) < radius ** 2


@torch.no_grad()
def run_protocol(adapter, target, proto=Protocol(), collect_frames=True):
    """1本の評価を回して、ステップごとの記録と、描画用のコマを返す。
    rows[t] は t 回目のステップを終えたあとの状態の記録。"""
    target = target.unsqueeze(0) if target.dim() == 3 else target
    h, w = target.shape[-2:]
    hole = damage_hole(h, w, proto.damage_center, proto.damage_radius, target.device)
    keep = set(SHEET_STEPS)

    state = adapter.reset()
    rows, frames = [], []
    cum = 0
    acc = torch.zeros(h, w, device=target.device)     # 前のコマからの発火の合計
    for t in range(proto.total_steps):
        if t == proto.damage_at:
            state = adapter.damage(state, hole)
        state, ev = adapter.step(state)
        fires, alive = ev["fires"], ev["alive"]
        rgba = adapter.rgba(state)
        n_fire = int(fires.sum().item())
        cum += n_fire
        rows.append(dict(
            t=t, phase=proto.phase(t),
            loss=float(((rgba - target) ** 2).mean().item()),
            fires=n_fire, alive=int(alive.sum().item()), cum_fires=cum,
        ))
        acc += fires[0, 0].float()
        if collect_frames and ((t + 1) % proto.frame_every == 0 or t in keep):
            span = proto.frame_every if (t + 1) % proto.frame_every == 0 else 1
            frames.append(dict(
                t=t, phase=proto.phase(t),
                rgba=rgba[0].float().cpu().numpy(),
                fire_rate=(acc / span).clamp(max=1).cpu().numpy(),
                gif=(t + 1) % proto.frame_every == 0,
            ))
        if (t + 1) % proto.frame_every == 0:
            acc.zero_()
    return rows, frames


def _first_hit(loss, fires, start, end, tau):
    idx = np.nonzero(loss[start:end] <= tau)[0]
    if len(idx) == 0:
        return None
    k = int(idx[0])
    return dict(steps=k + 1, fires=int(fires[start:start + k + 1].sum()))


def summarize(rows, proto=Protocol()):
    loss = np.array([r["loss"] for r in rows])
    fires = np.array([r["fires"] for r in rows], dtype=np.int64)
    alive = np.array([r["alive"] for r in rows], dtype=np.int64)

    def window(a, b):
        return dict(
            loss_mean=float(loss[a:b].mean()),
            loss_end=float(loss[b - 1]),
            fires_total=int(fires[a:b].sum()),
            fires_per_step=float(fires[a:b].mean()),
            fire_rate=float(fires[a:b].sum() / max(alive[a:b].sum(), 1)),   # 生きているセルあたり
            alive_mean=float(alive[a:b].mean()),
        )

    s = dict(
        protocol=proto.to_dict(),
        total_fires=int(fires.sum()),
        grow=window(0, proto.grow_end),
        maintain=window(proto.grow_end, proto.damage_at),
        recover=window(proto.damage_at, proto.total_steps),
        damage_loss_peak=float(loss[proto.damage_at:proto.damage_at + 20].max()),
    )
    # 成長：ロスが初めて τ 以下になったステップ数と、そこまでの総発火数
    s["grow_first_hit"] = {str(tau): _first_hit(loss, fires, 0, proto.damage_at, tau)
                           for tau in proto.thresholds}
    # 回復：傷のあと、ロスが τ 以下に戻るまでのステップ数と発火数
    s["recover_first_hit"] = {str(tau): _first_hit(loss, fires, proto.damage_at, proto.total_steps, tau)
                              for tau in proto.thresholds}
    # 回復（相対）：維持期の平均ロスの1.1倍以下に戻るまで
    ref = s["maintain"]["loss_mean"] * 1.1
    s["recover_to_maintain"] = _first_hit(loss, fires, proto.damage_at, proto.total_steps, ref)
    return s


def write_rows_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
