"""
Step 2 の評価のプロトコル（DESIGN.md 6章）。Step 1 のもの（common/protocol.py）は変えない。

    0〜400：学ぶ（予測器は0、σ は下限から）／400〜800：落ち着く／800：光源の周期を変える（位相はつなげたまま）／800〜1400：学び直す

- 指標はすべて区域ごと（波・ノイズ）。世界は n_worlds 個を並べて回し、セルと世界で平均する
- 周期を変えたあとの指標は、新しい波がその列に届いた時刻（world.column_period で見る）から測る
- 2a（dense=True）は全員が毎ステップ書き換える。引き金は数えるだけ（2b と同じ数え方）
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch

import world as W
from model import init_state, irreducible, nll


@dataclass
class Protocol2:
    total: int = 1400
    settle: tuple = (400, 800)        # 落ち着いた時期
    change_at: int = 800              # この時刻に、新しい周期で進んだ最初の位相が光源に出る
    P1: float = 12.0
    P2: float = 24.0
    n_worlds: int = 4
    relearn: int = 300                # 学び直しの窓（新しい波が届いてから）
    frame_steps: tuple = (0, 5, 20, 100, 399, 799, 810, 850, 900, 950, 1100, 1399)

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def run_protocol2(cell, wcfg, proto, eval_seed, dense=True, device="cpu", frames=False):
    """1回の評価。戻り値：まとめ（dict）、時間ごとの記録（list of dict）、絵のコマ（frames=True のとき）"""
    g = torch.Generator(device=device).manual_seed(eval_seed)
    ws = W.reset(wcfg, proto.P1, n=proto.n_worlds, generator=g, device=device)
    H = Wd = wcfg.size
    st = init_state(proto.n_worlds, H, Wd, device)
    zone = W.zone_mask(wcfg, device)                       # (H,W) ノイズの区域
    wave = ~zone
    T = proto.total
    # セルごとの記録（評価は短いので全部持つ）：誤差の2乗、NLL、z²、引き金、書き換え、σ
    E2 = torch.zeros(T, proto.n_worlds, H, Wd, device=device)
    NL = torch.zeros_like(E2)
    Z2 = torch.zeros_like(E2)
    TR = torch.zeros_like(E2, dtype=torch.bool)
    WR = torch.zeros_like(E2, dtype=torch.bool)
    SG = torch.zeros_like(E2)
    arrival = torch.full((Wd,), -1, dtype=torch.long)
    rows, pics = [], []
    for t in range(T):
        if t == proto.change_at - 1:
            ws = W.set_period(ws, proto.P2)                # 次の advance で新しい周期の位相が光源に出る（時刻 change_at）
        sense = W.sense(ws, wcfg, g)
        st, rec = cell.step(st, sense, dense=dense)
        E2[t] = rec["e"][:, 0] ** 2
        NL[t] = nll(rec["e"], rec["sigma"], detach_sigma=False)[:, 0]
        Z2[t] = rec["z2"][:, 0]
        TR[t] = rec["trig"][:, 0]
        WR[t] = rec["write"][:, 0]
        SG[t] = rec["sigma"][:, 0]
        cp = W.column_period(ws, wcfg)[0]                  # (W,) 世界0の列ごとの周期（どの世界も同じ時刻に変える）
        newly = (cp == proto.P2).cpu() & (arrival < 0) & (t >= proto.change_at)
        arrival[newly] = t
        rows.append(dict(t=t, nll_wave=float(NL[t][:, wave].mean()), nll_noise=float(NL[t][:, zone].mean()),
                         z2_wave=float(Z2[t][:, wave].mean()), z2_noise=float(Z2[t][:, zone].mean()),
                         trig_wave=float(TR[t][:, wave].float().mean()), trig_noise=float(TR[t][:, zone].float().mean()),
                         write_wave=float(WR[t][:, wave].float().mean()),
                         sigma_wave=float(SG[t][:, wave].median()), sigma_noise=float(SG[t][:, zone].median()),
                         eta=float(rec["eta"].mean()), g=float(rec["g"].mean())))
        if frames and t in proto.frame_steps:
            pics.append(dict(t=t, sense=sense[0, 0].cpu().numpy(), z=Z2[t][0].cpu().numpy(),
                             sigma=SG[t][0].cpu().numpy(), trig=TR[t][0].cpu().numpy()))
        ws = W.advance(ws)
    return summarize2(E2, NL, Z2, TR, WR, SG, arrival, wave, zone, proto, wcfg), rows, pics


def summarize2(E2, NL, Z2, TR, WR, SG, arrival, wave, zone, proto, wcfg):
    a, b = proto.settle
    out = dict(P1=proto.P1, P2=proto.P2)
    irr = irreducible(wcfg.noise, proto.P1)
    out["irreducible"] = irr
    out["nll_irreducible"] = 0.5 + float(np.log(irr))       # 当てられない誤差の正規分布の NLL（σ が合っているとき）
    for name, m in (("wave", wave), ("noise", zone)):
        out[f"settle_nll_{name}"] = float(NL[a:b][:, :, m].mean())
        out[f"settle_z2_{name}"] = float(Z2[a:b][:, :, m].mean())
        out[f"settle_err_{name}"] = float(E2[a:b][:, :, m].mean().sqrt())
        out[f"settle_trig_{name}"] = float(TR[a:b][:, :, m].float().mean())
        out[f"settle_write_{name}"] = float(WR[a:b][:, :, m].float().mean())
        out[f"sigma_median_{name}"] = float(SG[b - 1][:, m].median())
    out["settle_err_ratio_wave"] = out["settle_err_wave"] / irr
    out["sigma_ratio"] = out["sigma_median_noise"] / out["sigma_median_wave"]
    out["learn_trig_wave"] = float(TR[:a][:, :, wave].float().sum() / (TR.shape[1] * int(wave.sum())))
    # 周期を変えたあと（列ごとに、新しい波が届いた時刻から）
    H, Wd = wave.shape
    cols = []
    for x in range(Wd):
        t0 = int(arrival[x])
        rows_w = wave[:, x]                                  # この列の波のセル
        if t0 < 0 or t0 + proto.relearn > E2.shape[0] or not bool(rows_w.any()):
            continue
        col = lambda A_: A_[..., x][..., rows_w]            # 最後の軸が列 x、その中の波の行
        pre_nll = float(col(NL[a:b]).mean())
        pre_err = float(col(E2[a:b]).mean().sqrt())
        post = slice(t0 + proto.relearn - 50, t0 + proto.relearn)
        tr = col(TR)                                          # (T, n_worlds, 波の行)
        cols.append(dict(
            x=x, arrival=t0,
            react_col=float(tr[t0:t0 + 10].any(0).any(-1).float().mean()),      # この列で10歩以内に引き金（世界で平均）
            react_cell=float(tr[t0:t0 + 10].any(0).float().mean()),
            nll_back=float(col(NL[post]).mean()) <= pre_nll + 0.1,
            err_back=float(col(E2[post]).mean().sqrt()) <= 2 * pre_err,
            gaveup=float((col(SG[t0 + proto.relearn - 1]) > 2 * col(SG[proto.change_at - 1])).float().mean()),
            quiet_after=float(tr[t0 + proto.relearn - 100:t0 + proto.relearn].float().mean()),
            relearn_trig=float(tr[t0:t0 + proto.relearn].float().sum(0).mean())))
    if cols:
        out["react_cols"] = float(np.mean([c["react_col"] >= 0.5 for c in cols]))   # 世界の半分以上で反応した列の割合
        out["react_cell"] = float(np.mean([c["react_cell"] for c in cols]))
        out["nll_back_cols"] = float(np.mean([c["nll_back"] for c in cols]))
        out["err_back_cols"] = float(np.mean([c["err_back"] for c in cols]))
        out["gaveup"] = float(np.mean([c["gaveup"] for c in cols]))
        out["quiet_after_wave"] = float(np.mean([c["quiet_after"] for c in cols]))
        left = [c["relearn_trig"] for c in cols if 1 <= c["x"] <= 8]
        right = [c["relearn_trig"] for c in cols if Wd - 8 <= c["x"] <= Wd - 1]
        out["relearn_trig_left"] = float(np.mean(left)) if left else None
        out["relearn_trig_right"] = float(np.mean(right)) if right else None
        out["left_right_ratio"] = (out["relearn_trig_right"] / out["relearn_trig_left"]
                                   if left and right and out["relearn_trig_left"] > 0 else None)
    out["after_trig_noise"] = float(TR[proto.change_at:][:, :, zone].float().mean())
    out["columns"] = cols
    return out


def judge_2a(s):
    """2a の合格（DESIGN.md 7.1。数字は 2026-09-29 に走らせる前に決めた）：落ち着いた時期（400〜800）だけで判定する"""
    checks = dict(
        sigma_wave=s["sigma_median_wave"] <= 0.015,
        sigma_noise=0.25 <= s["sigma_median_noise"] <= 0.33,
        sigma_ratio=s["sigma_ratio"] >= 5,
        calib_wave=0.5 <= s["settle_z2_wave"] <= 1.5,
        calib_noise=0.5 <= s["settle_z2_noise"] <= 1.5,
        nll_wave=s["settle_nll_wave"] <= s["nll_irreducible"] + 0.2,
    )
    return all(checks.values()), checks
