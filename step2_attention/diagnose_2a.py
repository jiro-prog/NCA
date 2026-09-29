"""
2a の診断（2026-09-29）：DNA ありの誤差が「2つ前までの線形予測で当てられない誤差」を下回った。
2a は全員が毎ステップ書き換えるので、DNA は係数（とくに切片 b）を毎回書き直して、予測器の形を飛び越えて自分で予測できる。
何の情報を使っているかを、学び直さずに評価のときだけ知覚を隠して切り分ける。

    .venv/Scripts/python.exe step2_attention/diagnose_2a.py step2_attention/runs/2a_seed0

- 列ごとの落ち着いた誤差：左隣のいない x=0 だけ悪ければ、左隣ののぞき見（左隣の3ステップ前の値＝自分の1ステップ先）
- 隠す1：近所8か所の最近のセンサーの値を0.5（真ん中）にする
- 隠す2：自分の古い値（s(t−2)・s(t−3)）を0.5にする（予測器が使う s(t)・s(t−1) は残す）
- 隠す3：近所8か所のすべてのチャンネル（係数・σ・濃度・隠れ・最近の値）を自分の値にする（近所から新しいことがわからない）
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, N_HIST, init_state, irreducible  # noqa: E402

CH_HIST0 = 3 + 1 + 1 + 4          # 近所のチャンネルのうち、最近の値が始まる番号（係数3・log σ・濃度・隠れ4 のあと）


def masked_perceive(mode):
    def perceive(self, nbr, self_extra):
        B, C, H, Wd = nbr.shape
        p = F.unfold(nbr, 3, padding=1).view(B, C, 9, H, Wd).clone()
        if mode == "hide_nbr_sense":
            for k in range(N_HIST):
                p[:, CH_HIST0 + k, [0, 1, 2, 3, 5, 6, 7, 8]] = 0.5
        elif mode == "hide_own_old":
            for k in (2, 3):
                p[:, CH_HIST0 + k, 4] = 0.5
        elif mode == "hide_all_nbr":
            p[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = p[:, :, 4:5]
        return torch.cat([p.view(B, C * 9, H, Wd), self_extra], 1)
    return perceive


@torch.no_grad()
def steady_by_column(cell, wcfg, P=12.0, seed=1000, n=4, dev="cuda", T=800, a=400):
    g = torch.Generator(device=dev).manual_seed(seed)
    ws = W.reset(wcfg, P, n=n, generator=g, device=dev)
    st = init_state(n, wcfg.size, wcfg.size, dev)
    wave = ~W.zone_mask(wcfg, dev)
    E2 = torch.zeros(wcfg.size, device=dev)
    cnt = torch.zeros(wcfg.size, device=dev)
    for t in range(T):
        st, rec = cell.step(st, W.sense(ws, wcfg, g), dense=True)
        if t >= a:
            e2 = (rec["e"][:, 0] ** 2) * wave
            E2 += e2.sum((0, 1))
            cnt += wave.float().sum(0) * n
        ws = W.advance(ws)
    return (E2 / cnt).sqrt().cpu().numpy() / irreducible(wcfg.noise, P)


def main():
    run = Path(sys.argv[1])
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    wcfg = W.WorldConfig(**cfg["world"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dna = AttentionCell(dna=True).to(dev)
    dna.load_state_dict(torch.load(run / "model.pt", map_location=dev))
    ref = AttentionCell(dna=False).to(dev)
    orig = AttentionCell.perceive
    rows = []
    for name, cell, mode in [("DNA なし（毎ステップ最小二乗）", ref, None), ("DNA あり", dna, None),
                             ("DNA あり・近所の最近の値を隠す", dna, "hide_nbr_sense"),
                             ("DNA あり・自分の古い値（s(t−2)・s(t−3)）を隠す", dna, "hide_own_old"),
                             ("DNA あり・近所を全部隠す（自分と同じに見せる）", dna, "hide_all_nbr")]:
        AttentionCell.perceive = masked_perceive(mode) if mode else orig
        col = steady_by_column(cell, wcfg, dev=dev)
        rows.append((name, col))
    AttentionCell.perceive = orig
    out = ["# 2a の診断：何の情報で、当てられない誤差を下回ったか", "",
           f"`{run.name}`、周期12、落ち着いた時期（400〜800）の波の誤差 ÷ 当てられない誤差（2つ前までの線形予測）。世界4つ、評価の乱数1000。", "",
           "| 条件 | 全体 | x=0（左隣なし） | x=1 | x=2〜30 | x=31（右隣なし） |", "|---|---:|---:|---:|---:|---:|"]
    for name, col in rows:
        out.append(f"| {name} | {col.mean():.2f} | {col[0]:.2f} | {col[1]:.2f} | {col[2:31].mean():.2f} | {col[31]:.2f} |")
    text = "\n".join(out) + "\n"
    (run / "diagnose.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
