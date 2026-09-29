"""
Step 2 の学習と評価（DESIGN.md 5章・6章）。いまは 2a（全員が毎ステップ書き換える）。

    .venv/Scripts/python.exe step2_attention/train.py --seed 0 --name 2a_seed0
    .venv/Scripts/python.exe step2_attention/train.py --eval-only step2_attention/runs/2a_seed0

- DNA（全セル共通の MLP）を、ロールアウトを通した誤差逆伝播で学ぶ。損失は NLL の平均（σ は固定値として扱う）
- プール：セルの状態と世界の状態を持ち、続きから回す。毎回、バッチの一部を新しい世界（予測器0から）に入れ替え、
  一部はロールアウトの途中で周期を変える（この予測器に見える変化だけ。DESIGN.md 1章）
- 守り（Step 1 から持ち越し）：勾配に無限大や nan が混ざった回は更新を飛ばし、回数を記録する。勾配はパラメータごとに正規化する
- 評価：DNA ありと、DNA なし（η=1、g=1、Δw=0 の密な最小二乗）を、同じ世界で並べる。周期の変化は 12→24・12→20・12→36・24→12
- 封印テストのシード（9000以上）は拒む（sealed.py）
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    sys.path.insert(0, str(p))

from common.seeding import seed_everything  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, Physio2, State2, init_state, nll  # noqa: E402
from protocol import Protocol2, judge_2a, run_protocol2  # noqa: E402
from sealed import check_dev_seed  # noqa: E402

EVAL_CHANGES = ((12.0, 24.0), (12.0, 20.0), (12.0, 36.0), (24.0, 12.0))


def git_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+未コミットの変更あり" if dirty else "")
    except OSError:
        return None


class Pool:
    """セルの状態と世界の状態を CPU に持つ"""

    def __init__(self, n, wcfg, gen):
        self.wcfg = wcfg
        self.cell = init_state(n, wcfg.size, wcfg.size)
        self.world = W.reset(wcfg, W.sample_periods(n, gen), n=n, generator=gen)

    def take(self, idx, dev):
        return State2(*(t[idx].to(dev) for t in self.cell)), self.world.index(idx).to(dev)

    def store(self, idx, cell, world):
        for dst, src in zip(self.cell, cell):
            dst[idx] = src.detach().cpu()
        self.world.phase[idx] = world.phase.cpu()
        self.world.period[idx] = world.period.cpu()
        self.world.P[idx] = world.P.cpu()


def fresh(cell, world, j, wcfg, gen, dev):
    """バッチの j 番目を、新しい世界（周期を選び直す）と初めの状態（予測器0、σ は下限）に入れ替える"""
    new_c = init_state(1, wcfg.size, wcfg.size, dev)
    cell = State2(*(torch.cat([t[:j], n_, t[j + 1:]]) for t, n_ in zip(cell, new_c)))
    P = W.sample_periods(1, gen)
    nw = W.reset(wcfg, P, n=1, generator=gen).to(dev)          # 乱数は CPU で引いてから GPU へ
    world = W.WorldState(torch.cat([world.phase[:j], nw.phase, world.phase[j + 1:]]),
                         torch.cat([world.period[:j], nw.period, world.period[j + 1:]]),
                         torch.cat([world.P[:j], nw.P, world.P[j + 1:]]))
    return cell, world


def train(args, run, dev):
    wcfg = W.WorldConfig(noise=args.noise)
    gen_cpu = torch.Generator().manual_seed(args.seed + 11)            # プールの抽選、周期、入れ替え、変える時刻
    gen_world = torch.Generator(device=dev).manual_seed(args.seed + 12)  # 世界のノイズ
    pool = Pool(args.pool, wcfg, gen_cpu)
    model = AttentionCell(dna=True).to(dev)
    model.use_checkpoint = bool(args.checkpoint)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [args.lr_drop_at], 0.1)
    zone = W.zone_mask(wcfg, dev)
    wave = ~zone
    log_f = open(run / "train_log.csv", "w", newline="", encoding="utf-8")
    cols = ["iter", "steps", "loss", "nll_wave", "nll_noise", "z2_wave", "z2_noise", "trig_wave", "trig_noise",
            "eta", "g", "dw_abs", "skipped", "lr", "seconds"]
    wr = csv.DictWriter(log_f, fieldnames=cols)
    wr.writeheader()
    (run / "checkpoints").mkdir(exist_ok=True)
    skipped = 0
    t0 = time.time()
    for it in range(args.iters):
        idx = torch.randperm(args.pool, generator=gen_cpu)[:args.batch]
        cell, world = pool.take(idx, dev)
        for j in range(args.fresh_per_batch):                         # 初めから学ぶサンプル
            cell, world = fresh(cell, world, j, wcfg, gen_cpu, dev)
        T = int(torch.randint(args.rollout_min, args.rollout_max + 1, (1,), generator=gen_cpu))
        # 周期を途中で変えるサンプル：変える時刻と行き先（見える組だけ）
        change_j = list(range(args.fresh_per_batch, args.fresh_per_batch + args.change_per_batch))
        change_t = {j: int(torch.randint(1, T, (1,), generator=gen_cpu)) for j in change_j}
        change_P = {j: float(W.sample_visible_change(world.P[j:j + 1].cpu(), gen_cpu)[0]) for j in change_j}
        loss = 0.0
        acc = dict(nll_wave=0.0, nll_noise=0.0, z2_wave=0.0, z2_noise=0.0, trig_wave=0.0, trig_noise=0.0, eta=0.0, g=0.0)
        dw_abs = 0.0
        for t in range(T):
            for j in change_j:
                if t == change_t[j]:
                    mask = torch.zeros(args.batch, dtype=torch.bool, device=dev)
                    mask[j] = True
                    world = W.set_period(world, change_P[j], mask=mask)
            sense = W.sense(world, wcfg, gen_world)
            w_before = cell.w
            cell, rec = model.step(cell, sense, dense=bool(args.dense))
            l = nll(rec["e"], rec["sigma"], detach_sigma=True)
            loss = loss + l.mean()
            with torch.no_grad():
                acc["nll_wave"] += float(l[:, 0][:, wave].mean())
                acc["nll_noise"] += float(l[:, 0][:, zone].mean())
                acc["z2_wave"] += float(rec["z2"][:, 0][:, wave].mean())
                acc["z2_noise"] += float(rec["z2"][:, 0][:, zone].mean())
                acc["trig_wave"] += float(rec["trig"][:, 0][:, wave].float().mean())
                acc["trig_noise"] += float(rec["trig"][:, 0][:, zone].float().mean())
                acc["eta"] += float(rec["eta"].mean())
                acc["g"] += float(rec["g"].mean())
                dw_abs += float((cell.w - w_before).abs().mean())
            world = W.advance(world)
        loss = loss / T
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            for p in model.parameters():                               # パラメータごとに勾配を正規化（Step 1 と同じ）
                if p.grad is not None:
                    p.grad /= p.grad.norm() + 1e-8
            opt.step()
        else:
            skipped += 1
        sched.step()
        pool.store(idx, cell, world)
        row = dict(iter=it, steps=T, loss=float(loss), skipped=skipped, lr=sched.get_last_lr()[0],
                   seconds=round(time.time() - t0, 1), dw_abs=dw_abs / T, **{k: v / T for k, v in acc.items()})
        wr.writerow(row)
        if it % 50 == 0 or it == args.iters - 1:
            log_f.flush()
            print(f"iter {it:5d}  損失 {row['loss']:.3f}  NLL 波 {row['nll_wave']:.3f}／ノイズ {row['nll_noise']:.3f}  "
                  f"z² 波 {row['z2_wave']:.2f}／ノイズ {row['z2_noise']:.2f}  η {row['eta']:.3f} g {row['g']:.4f}  "
                  f"飛ばした {skipped}  ({row['seconds']:.0f}秒)", flush=True)
        if args.save_every and (it + 1) % args.save_every == 0:
            torch.save(model.state_dict(), run / "checkpoints" / f"iter{it + 1}.pt")
    log_f.close()
    return model, dict(skipped_updates=skipped, sec_per_iter=round((time.time() - t0) / max(args.iters, 1), 3))


def save_pictures(pics, rows, out_dir, title):
    """コマ（センサー、z、σ、引き金）と、NLL・z²・引き金の推移"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.viz import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    if pics:
        fig, axes = plt.subplots(4, len(pics), figsize=(1.35 * len(pics), 6.0), facecolor=SURFACE)
        for k, p in enumerate(pics):
            for r_, (arr, cmap, vmin, vmax, lab) in enumerate([
                    (p["sense"], "gray", 0, 1, "センサー"), (np.sqrt(p["z"]), "magma", 0, 4, "|誤差|/σ"),
                    (np.log10(p["sigma"]), "viridis", np.log10(0.005), np.log10(0.4), "log σ"),
                    (p["trig"].astype(float), "Oranges", 0, 1, "引き金")]):
                ax = axes[r_, k]
                ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if r_ == 0:
                    ax.set_title(f"t={p['t']}", fontsize=8, color=INK_2)
                if k == 0:
                    ax.set_ylabel(lab, fontsize=8, color=INK_2)
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(out_dir / "sheet.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
    t = [r["t"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True, facecolor=SURFACE)
    for ax, keys, name in [(axes[0], ("nll_wave", "nll_noise"), "NLL（セルの平均）"),
                           (axes[1], ("z2_wave", "z2_noise"), "z² の平均（自信度が合えば1）"),
                           (axes[2], ("trig_wave", "trig_noise"), "引き金の率（2a では数えるだけ）")]:
        ax.set_facecolor(SURFACE)
        for c_, (k, lab) in enumerate(zip(keys, ("波", "ノイズ"))):
            ax.plot(t, [r[k] for r in rows], color=SERIES[c_], linewidth=1.2, label=lab)
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(AXIS)
        for x0 in (400, 800):
            ax.axvline(x0, color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[2].set_xlabel("ステップ（800 で周期を変える）", color=INK_2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def evaluate(model, wcfg, out_dir, title, eval_seed, dev, dense=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for P1, P2 in EVAL_CHANGES:
        proto = Protocol2(P1=P1, P2=P2)
        s, rows, pics = run_protocol2(model, wcfg, proto, eval_seed, dense=dense, device=dev,
                                      frames=(P1, P2) == EVAL_CHANGES[0])
        key = f"{P1:g}to{P2:g}"
        cols = s.pop("columns")
        res[key] = s
        with open(out_dir / f"columns_{key}.csv", "w", newline="", encoding="utf-8") as f:
            if cols:
                w_ = csv.DictWriter(f, fieldnames=list(cols[0].keys()))
                w_.writeheader()
                w_.writerows(cols)
        with open(out_dir / f"steps_{key}.csv", "w", newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w_.writeheader()
            w_.writerows(rows)
        if (P1, P2) == EVAL_CHANGES[0]:
            save_pictures(pics, rows, out_dir, f"{title}（周期 {P1:g}→{P2:g}）")
    ok, checks = judge_2a(res["12to24"])
    res["judge_2a"] = dict(passed=ok, checks=checks)
    (out_dir / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def brief(res):
    s = res["12to24"]
    return (f"落ち着いた時期：NLL 波 {s['settle_nll_wave']:.3f}（当てられない誤差なら {s['nll_irreducible']:.3f}）、"
            f"z² 波 {s['settle_z2_wave']:.2f}／ノイズ {s['settle_z2_noise']:.2f}、σ 波 {s['sigma_median_wave']:.4f}／ノイズ {s['sigma_median_noise']:.3f}、"
            f"誤差 {s['settle_err_ratio_wave']:.2f}倍 → 2a の合格 {'○' if res['judge_2a']['passed'] else '×'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4, help="DNA は良い書き換え（DNA なし）から始まるので小さく。1e-3 は試運転で3回のうちに崩れた（台帳）")
    ap.add_argument("--lr-drop-at", type=int, default=1000, help="この回で学習率を1/10に")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--rollout-min", type=int, default=96)
    ap.add_argument("--rollout-max", type=int, default=160)
    ap.add_argument("--fresh-per-batch", type=int, default=1, help="毎回、初めから学ぶサンプルの数")
    ap.add_argument("--change-per-batch", type=int, default=2, help="毎回、途中で周期を変えるサンプルの数")
    ap.add_argument("--dense", type=int, default=1, help="1 なら 2a（全員が毎ステップ書き換える）")
    ap.add_argument("--noise", type=float, default=W.WorldConfig().noise)
    ap.add_argument("--checkpoint", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-only", default=None)
    ap.add_argument("--eval-seed", type=int, default=1000)
    ap.add_argument("--device", default=None, help="cpu か cuda（既定は使えれば cuda）。試運転を CPU で回すとき")
    args = ap.parse_args()
    check_dev_seed(args.seed, "学習のシード")
    check_dev_seed(args.eval_seed, "評価のシード")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        run = Path(args.eval_only)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        wcfg = W.WorldConfig(**cfg["world"])
        model = AttentionCell(dna=True).to(dev)
        model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
    else:
        run = HERE / "runs" / (args.name or f"2a_seed{args.seed}")
        run.mkdir(parents=True, exist_ok=True)
        wcfg = W.WorldConfig(noise=args.noise)
        cfg = dict(vars(args), stage="2a" if args.dense else "2b", physio=Physio2.to_dict(), world=wcfg.to_dict(),
                   git=git_commit(), torch=torch.__version__,
                   device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", status="training")
        cfg.pop("eval_only")
        (run / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        seed_everything(args.seed)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, info = train(args, run, dev)
        cfg.update(info, status="done", train_seconds=round(time.time() - t0, 1),
                   peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None)
        (run / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), run / "model.pt")
        print(f"学習 {cfg['train_seconds']} 秒（1回 {info['sec_per_iter']} 秒）、VRAM最大 {cfg['peak_vram_mb']} MB、"
              f"更新を飛ばした回数 {info['skipped_updates']}", flush=True)

    model.eval()
    res = evaluate(model, wcfg, run / "eval", f"2a seed{cfg['seed']} DNA あり", args.eval_seed, dev, dense=True)
    print("DNA あり：", brief(res), flush=True)
    ref = AttentionCell(dna=False).to(dev)
    res_ref = evaluate(ref, wcfg, run / "eval_ref", "2a DNA なし（密な最小二乗）", args.eval_seed, dev, dense=True)
    print("DNA なし：", brief(res_ref), flush=True)
    d = res["12to24"]["settle_nll_wave"] - res_ref["12to24"]["settle_nll_wave"]
    print(f"波の NLL の差（DNA あり − なし、落ち着いた時期）：{d:+.3f} ナット", flush=True)
    print("出力:", run)


if __name__ == "__main__":
    main()
