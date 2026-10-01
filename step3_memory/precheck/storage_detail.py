"""
しまうタイミングの診断の補い（読み方は事前に決めていない。判定には使わない）：(し1) の重複が、どのパターンで、切り替えを見逃したセルに偏っているか。

    .venv/Scripts/python.exe step3_memory/precheck/storage_detail.py [--device cuda]

storage_check.py と同じ回し方（眠る段のそばで保存先の帳簿だけ）で、(し1) だけを帳簿づけし、あわせて各切り替えへの反応（新しい波が届いてから300ステップの中で驚いて発火したか）を数える。
出力：step3_memory/runs/precheck_storage/detail.md
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import storage_check as SC  # noqa: E402
from storage_check import C, E, W, Physio2, pred_gap  # noqa: E402


@torch.no_grad()
def run(wcfg, seed, start, dev, plan=E.Plan()):
    B = plan.n_per * len(plan.groups)
    S_ = wcfg.size
    per = {name: torch.tensor([grp[i] for grp in plan.groups for _ in range(plan.n_per)], dtype=torch.float64) for i, name in enumerate("ABC")}
    sched = [per[p] for p in plan.seq]
    T = len(plan.seq) * plan.dwell
    g = torch.Generator(device=dev).manual_seed(seed)
    if start == "adult":
        core, ws = E.adult_core(wcfg, sched[0], g, dev)
        st = C.wrap(core)
    else:
        ws = W.reset(wcfg, sched[0].to(dev), n=B, generator=g, device=dev)
        st = C.init_state(B, S_, S_, dev)
    cell = C.Cell3(sleep=True, device=dev)
    book = SC.Book(B, S_, S_, dev)
    n_sw = len(plan.seq) - 1
    react = torch.zeros(n_sw, B, S_, S_, dtype=torch.bool, device=dev)
    xs = torch.arange(S_, device=dev)
    for t in range(T):
        j = t // plan.dwell
        if t % plan.dwell == plan.dwell - 1 and j + 1 < len(plan.seq):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        st, rec = cell.step(st, sense)
        book.tick()
        sl = rec["sleep"][:, 0]
        book.store(sl & (st.k[:, 0] >= 2), st.core.w, st.RL, st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0])
        if j >= 1:
            rel = (t - (j * plan.dwell + xs * wcfg.lag)).view(1, 1, S_)
            react[j - 1] |= rec["surprise"][:, 0] & (rel >= 0) & (rel < plan.window)
        ws = W.advance(ws)
    wave = ~W.zone_mask(wcfg, dev)
    first = 1 if start == "adult" else 0
    rows = []
    for gi, grp in enumerate(plan.groups):
        sl_ = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
        occ = (book.n[sl_] > 0)[:, :, wave]
        slots = book.slot[sl_].permute(0, 1, 3, 4, 2)[:, :, wave]
        gaps = torch.stack([pred_gap(slots, P) for P in grp], -1)
        best_gap, best_p = gaps.min(-1)
        hit = occ & (best_gap <= SC.MATCH)
        dup_cell = torch.zeros(occ.shape[0], occ.shape[2], dtype=torch.bool, device=dev)
        by_p = []
        for p in range(3):
            k_p = (hit & (best_p == p)).sum(1)
            by_p.append(int((k_p >= 2).sum()))
            dup_cell |= k_p >= 2
        missed = (~react[first:, sl_][:, :, wave]).any(0)                     # 判定する切り替えのどれかを見逃した
        rows.append(dict(group="/".join(f"{p:g}" for p in grp), dup_by_pattern=by_p, dup_cells=int(dup_cell.sum()),
                         missed_cells=int(missed.sum()), dup_and_missed=int((dup_cell & missed).sum()), n=int(dup_cell.numel())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    L = ["# (し1) の重複の中身（しまうタイミングの診断の補い。読み方は事前に決めていない）", "",
         "| 始め方 | 評価の乱数 | 組 | 重複のあるセル | うち A／B／C で重複 | 切り替えを見逃したことのあるセル | 重複かつ見逃したことのあるセル |", "|---|---:|---|---:|---|---:|---:|"]
    for start in ("adult", "grow"):
        for seed in (300, 301):
            for r in run(wcfg, seed, start, args.device):
                L.append(f"| {'大人' if start == 'adult' else '育ち'} | {seed} | {r['group']} | {r['dup_cells']} | {'／'.join(map(str, r['dup_by_pattern']))} | "
                         f"{r['missed_cells']} | {r['dup_and_missed']} |")
            print(start, seed, "done", flush=True)
    out = SC.ROOT / "step3_memory" / "runs" / "precheck_storage" / "detail.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
