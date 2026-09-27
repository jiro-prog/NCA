"""
元祖NCA（nca.py）を改造せずに外から包んで、シード固定・発火の記録つきで学習と評価を行う。

使い方（プロジェクトの直下で）:
    .venv/Scripts/python.exe baseline/run_baseline.py                   # 学習4000回 → 評価
    .venv/Scripts/python.exe baseline/run_baseline.py --iters 30 --name smoke
    .venv/Scripts/python.exe baseline/run_baseline.py --fire-rate 0.25  # ヌル対照：発火率を変えて学習し直す
    .venv/Scripts/python.exe baseline/run_baseline.py --period 2        # ヌル対照：kステップごとに全員が発火
    .venv/Scripts/python.exe baseline/run_baseline.py --eval-only baseline/runs/seed0_dice0.5

包み方:
- nca.NCA.fire_mask と nca.NCA.forward をクラスごと差し替えて、発火とロスを記録する。
  サイコロは nca.py の fire_mask をそのまま呼ぶので、乱数の使い方は変わらない（生死判定は乱数を使わない）
- サイコロの確率は nca.FIRE_RATE を書き換えて変える（fire_mask が実行時に読むため）

出力（baseline/runs/<名前>/）:
    config.json       設定、バージョン、gitのコミット、所要時間、VRAMの最大使用量
    train_log.csv     学習の各回：ロス、発火数、生きているセル×ステップ数、発火率
    nca.pt            学習した重み
    eval/             3期の評価（steps.csv, summary.json, sheet.png, curves.png, eval.gif）
    eval_timerstop/   ヌル対照「時限停止」：成長期の終わりで全員の発火を止めたときの評価
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))

from common.seeding import seed_everything  # noqa: E402
import torch  # noqa: E402

import nca  # noqa: E402
from common import viz  # noqa: E402
from common.protocol import SHEET_STEPS, Protocol, run_protocol, summarize, write_rows_csv  # noqa: E402

ORIG_FIRE_MASK = nca.NCA.fire_mask
ORIG_FORWARD = nca.NCA.forward


# ---------------------------------------------------------------
# 発火の決め方（ヌル対照を含む）。policy(x, t) -> (B,1,H,W) の 0/1
# ---------------------------------------------------------------
def make_policy(model, period=None, stop_at=None):
    if period is None:
        def base(x, t):                         # nca.py のサイコロそのもの（nca.FIRE_RATE を読む）
            return ORIG_FIRE_MASK(model, x)
    else:
        def base(x, t):                         # k ステップごとに全員が発火（一定周期の時計）
            return torch.full_like(x[:, :1], float(t % period == 0))
    if stop_at is None:
        return base

    def stopped(x, t):                          # 時限停止：stop_at 以降は誰も発火しない
        return base(x, t) if t < stop_at else torch.zeros_like(x[:, :1])
    return stopped


# ---------------------------------------------------------------
# 学習中の記録
# ---------------------------------------------------------------
class Recorder:
    """学習の各回（1回のロールアウト）について、ロスと発火を CSV に書く"""

    def __init__(self, target, period, log_path):
        self.target, self.period = target, period
        self.t, self.it = 0, 0
        self.fires = self.alive = None
        self.f = open(log_path, "w", encoding="utf-8", newline="")
        self.f.write("iter,steps,loss,fires,alive_cell_steps,fire_rate,seconds\n")
        self.t0 = time.time()

    def install(self):
        rec = self

        def fire_mask(model, x):
            if rec.period is None:
                m = ORIG_FIRE_MASK(model, x)
            else:
                m = torch.full_like(x[:, :1], float(rec.t % rec.period == 0))
            with torch.no_grad():
                pre = nca.NCA.alive(x)
                rec.fires += ((m > 0) & pre).sum()
                rec.alive += pre.sum()
            rec.t += 1
            return m

        def forward(model, x, steps):
            rec.t = 0
            rec.fires = torch.zeros((), dtype=torch.long, device=x.device)
            rec.alive = torch.zeros((), dtype=torch.long, device=x.device)
            out = ORIG_FORWARD(model, x, steps)
            with torch.no_grad():
                loss = ((out[:, :4] - rec.target) ** 2).mean().item()
            fires, alive = int(rec.fires.item()), int(rec.alive.item())
            rec.f.write(f"{rec.it},{steps},{loss:.6g},{fires},{alive},"
                        f"{fires / max(alive, 1):.6f},{time.time() - rec.t0:.1f}\n")
            rec.it += 1
            if rec.it % 50 == 0:
                rec.f.flush()
            return out

        nca.NCA.fire_mask = fire_mask
        nca.NCA.forward = forward

    def uninstall(self):
        nca.NCA.fire_mask = ORIG_FIRE_MASK
        nca.NCA.forward = ORIG_FORWARD
        self.f.close()


# ---------------------------------------------------------------
# 評価
# ---------------------------------------------------------------
class BaselineAdapter:
    """nca.py のモデルを評価プロトコルで動かすためのアダプター（step は nca.py のものをそのまま使う）"""

    def __init__(self, model, h, w, device, policy):
        self.model, self.h, self.w, self.device, self.policy = model, h, w, device, policy
        self.t = 0

    def reset(self):
        self.t = 0
        return nca.make_seed(self.h, self.w).to(self.device)

    def step(self, x):
        pre = self.model.alive(x)
        m = self.policy(x, self.t)
        self.model.fire_mask = lambda _x: m         # この1ステップだけ、発火の決め方を差し替える
        try:
            x = self.model.step(x)
        finally:
            del self.model.fire_mask
        self.t += 1
        return x, {"fires": (m > 0) & pre, "alive": pre}

    def rgba(self, x):
        return x[:, :4]

    def damage(self, x, hole):
        x = x.clone()
        x[:, :, hole] = 0
        return x


def evaluate(model, h, w, device, out_dir, title, fire_rate, period=None, stop_at=None, eval_seed=1000):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(eval_seed)
    nca.FIRE_RATE = fire_rate
    proto = Protocol()
    adapter = BaselineAdapter(model, h, w, device, make_policy(model, period, stop_at))
    target = nca.make_target().to(device)
    model.eval()
    rows, frames = run_protocol(adapter, target, proto)
    summary = summarize(rows, proto)
    write_rows_csv(rows, out_dir / "steps.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    viz.save_sheet(frames, SHEET_STEPS, out_dir / "sheet.png", title=title)
    viz.save_curves(rows, proto, out_dir / "curves.png", title=title)
    viz.save_gif(frames, out_dir / "eval.gif", title=title)
    return summary


def git_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+未コミットの変更あり" if dirty else "")
    except OSError:
        return None


def brief(s):
    g, m, r = s["grow"], s["maintain"], s["recover"]
    return (f"成長 ロス末{g['loss_end']:.4f} 発火{g['fires_total']} | "
            f"維持 ロス平均{m['loss_mean']:.4f} 発火/ステップ{m['fires_per_step']:.1f} | "
            f"回復 ロス末{r['loss_end']:.4f} 発火{r['fires_total']} | 総発火{s['total_fires']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--fire-rate", type=float, default=0.5)
    ap.add_argument("--period", type=int, default=None, help="サイコロの代わりに k ステップごとに全員が発火")
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-only", default=None, help="学習済みの run フォルダを評価だけする")
    ap.add_argument("--eval-seed", type=int, default=1000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy_name = f"period{args.period}" if args.period else f"dice{args.fire_rate:g}"

    if args.eval_only:
        run = Path(args.eval_only)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        model = nca.NCA().to(dev)
        model.load_state_dict(torch.load(run / "nca.pt", map_location=dev))
        args.fire_rate, args.period = cfg["fire_rate"], cfg["period"]
        policy_name = cfg["policy"]
        h, w = nca.make_target().shape[-2:]
    else:
        run = ROOT / "baseline" / "runs" / (args.name or f"seed{args.seed}_{policy_name}")
        run.mkdir(parents=True, exist_ok=True)
        seed_everything(args.seed)
        nca.FIRE_RATE = args.fire_rate
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rec = Recorder(nca.make_target().to(dev), args.period, run / "train_log.csv")
        rec.install()
        t0 = time.time()
        try:
            model, h, w = nca.train(args.iters, dev)
        finally:
            rec.uninstall()
        cfg = dict(
            seed=args.seed, iters=args.iters, fire_rate=args.fire_rate, period=args.period,
            policy=policy_name, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            torch=torch.__version__, git=git_commit(), train_seconds=round(time.time() - t0, 1),
            peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None,
        )
        (run / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), run / "nca.pt")
        print(f"学習 {cfg['train_seconds']} 秒, VRAM最大 {cfg['peak_vram_mb']} MB")

    title = f"baseline {policy_name} seed{cfg['seed']}"
    s = evaluate(model, h, w, dev, run / "eval", title, args.fire_rate, args.period, eval_seed=args.eval_seed)
    print("評価        :", brief(s))
    s = evaluate(model, h, w, dev, run / "eval_timerstop", title + "（時限停止）", args.fire_rate, args.period,
                 stop_at=Protocol().grow_end, eval_seed=args.eval_seed)
    print("時限停止    :", brief(s))
    print("出力:", run)


if __name__ == "__main__":
    main()
