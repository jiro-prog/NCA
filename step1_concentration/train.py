"""
Step 1：濃度発火の花を学習して、3期のプロトコル（成長・維持・傷）で評価する。

使い方（プロジェクトの直下で）:
    .venv/Scripts/python.exe step1_concentration/train.py --lam 0.01
    .venv/Scripts/python.exe step1_concentration/train.py --iters 30 --name smoke
    .venv/Scripts/python.exe step1_concentration/train.py --eval-only step1_concentration/runs/<名前>

学習の手順は元祖（baseline/nca.py）と同じ：サンプルプール1024、バッチ8、最悪サンプルを種に戻す、
良いサンプル2つに穴を開ける、ステップ数64〜96、勾配をパラメータごとに正規化、学習率2e-3を2000回目で1/10。
違いはロスに「λ × 発火数」を足すことだけ（発火数はキャンバスのセル×ステップ数で割る）。

出力（step1_concentration/runs/<名前>/）:
    config.json, train_log.csv, model.pt
    eval_j0/     タイミングのずれなしの評価（steps.csv, summary.json, sheet.png, curves.png, eval.gif）
    eval_j0.1/   発火が確率0.1で1ステップ遅れる環境での評価
    eval_j0.3/   同じく確率0.3
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "baseline", HERE):
    sys.path.insert(0, str(p))

from common.seeding import seed_everything  # noqa: E402
import torch  # noqa: E402

from nca import PAD, make_target  # noqa: E402  目標の絵は元祖と同じものを使う
from common import viz  # noqa: E402
from common.protocol import SHEET_STEPS, Protocol, damage_hole, run_protocol, summarize, write_rows_csv  # noqa: E402
from model import ConcentrationNCA, Physio, State, damage_state, seed_state  # noqa: E402

POOL_SIZE = 1024
BATCH = 8
JITTERS = (0.0, 0.1, 0.3)


def git_commit():
    """今のコミットと、追跡中のファイルに未コミットの変更があるか"""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+未コミットの変更あり" if dirty else "")
    except OSError:
        return None


# ---------------------------------------------------------------
# プール（CPU に置く）
# ---------------------------------------------------------------
def pool_take(pool, idx, device):
    return State(*(t[idx].to(device) for t in pool))


def pool_store(pool, idx, s):
    for dst, src in zip(pool, s):
        dst[idx] = src.detach().cpu()


def replace_sample(s, j, new):
    s = State(*(t.clone() for t in s))
    for dst, src in zip(s, new):
        dst[j] = src[0]
    return s


# ---------------------------------------------------------------
# 学習
# ---------------------------------------------------------------
def train(args, run, dev):
    target = make_target().to(dev)
    _, h, w = target.shape
    model = ConcentrationNCA().to(dev)
    model.use_checkpoint = args.checkpoint
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], gamma=0.1)
    pool = seed_state(h, w, POOL_SIZE)

    log = open(run / "train_log.csv", "w", encoding="utf-8", newline="")
    log.write("iter,steps,loss,rgba_loss,fire_cost,fires,alive_cell_steps,fire_rate,seconds\n")
    t0 = time.time()
    for i in range(args.iters):
        idx = random.sample(range(POOL_SIZE), BATCH)
        s = pool_take(pool, idx, dev)

        with torch.no_grad():
            err = ((s.x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
            worst = int(err.argmax())
            s = replace_sample(s, worst, seed_state(h, w, 1, dev))     # いちばん出来の悪いサンプルを種に戻す
            for j in err.argsort()[:2].tolist():                        # 出来の良い2つに穴を開ける
                cy, cx = random.randint(PAD, h - PAD), random.randint(PAD, w - PAD)
                s = damage_state(s, damage_hole(h, w, (cy, cx), 8, dev), j)

        steps = random.randint(64, 96)
        out, fire_sum, n_fire, n_alive = model(s, steps)
        rgba_loss = ((out.x[:, :4] - target) ** 2).mean()
        fire_cost = fire_sum.sum() / (BATCH * h * w * steps)
        loss = rgba_loss + args.lam * fire_cost

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():            # 勾配を正規化して学習を安定させる（元祖と同じ）
            p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        pool_store(pool, idx, out)

        nf, na = int(n_fire.item()), int(n_alive.item())
        log.write(f"{i},{steps},{loss.item():.6g},{rgba_loss.item():.6g},{fire_cost.item():.6g},"
                  f"{nf},{na},{nf / max(na, 1):.6f},{time.time() - t0:.1f}\n")
        if i % 100 == 0 or i == args.iters - 1:
            log.flush()
            print(f"iter {i:5d}  rgba {rgba_loss.item():.5f}  発火率 {nf / max(na, 1):.3f}  "
                  f"({time.time() - t0:.0f}秒)", flush=True)
    log.close()
    return model, h, w


# ---------------------------------------------------------------
# 評価
# ---------------------------------------------------------------
class Step1Adapter:
    def __init__(self, model, h, w, device, jitter_p=0.0, gen_seed=1000):
        self.model, self.h, self.w, self.device, self.jitter_p = model, h, w, device, jitter_p
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(gen_seed)

    def reset(self):
        return seed_state(self.h, self.w, 1, self.device)

    def step(self, s):
        s, _, hard, pre = self.model.step(s, self.jitter_p, self.gen)
        n = pre.sum().clamp(min=1)
        extra = dict(d_mean=float((s.d * pre).sum() / n), c_mean=float((s.c * pre).sum() / n))
        return s, {"fires": hard, "alive": pre, "extra": extra}

    def rgba(self, s):
        return s.x[:, :4]

    def damage(self, s, hole):
        return damage_state(s, hole, 0)


def evaluate(model, h, w, dev, out_dir, title, jitter_p, eval_seed=1000, pictures=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(eval_seed)
    proto = Protocol()
    model.eval()
    rows, frames = run_protocol(Step1Adapter(model, h, w, dev, jitter_p, eval_seed),
                                make_target().to(dev), proto, collect_frames=pictures)
    summary = summarize(rows, proto)
    summary["jitter_p"] = jitter_p
    write_rows_csv(rows, out_dir / "steps.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    viz.save_curves(rows, proto, out_dir / "curves.png", title=title)
    if pictures:
        viz.save_sheet(frames, SHEET_STEPS, out_dir / "sheet.png", title=title)
        viz.save_gif(frames, out_dir / "eval.gif", title=title)
    return summary


def brief(s):
    g, m, r = s["grow"], s["maintain"], s["recover"]
    return (f"成長 ロス末{g['loss_end']:.4f} 発火{g['fires_total']} | "
            f"維持 ロス平均{m['loss_mean']:.4f} 発火/ステップ{m['fires_per_step']:.1f} | "
            f"回復 ロス末{r['loss_end']:.4f} 発火{r['fires_total']} | 総発火{s['total_fires']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=0.01, help="発火のコストの重み λ")
    ap.add_argument("--checkpoint", type=int, default=1, help="1 なら勾配チェックポイントでVRAMを節約する")
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-only", default=None)
    ap.add_argument("--eval-seed", type=int, default=1000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.eval_only:
        run = Path(args.eval_only)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        model = ConcentrationNCA().to(dev)
        model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
        h, w = make_target().shape[-2:]
    else:
        run = HERE / "runs" / (args.name or f"seed{args.seed}_lam{args.lam:g}")
        run.mkdir(parents=True, exist_ok=True)
        cfg = dict(seed=args.seed, iters=args.iters, lam=args.lam, checkpoint=bool(args.checkpoint),
                   physio=Physio.to_dict(), git=git_commit(), torch=torch.__version__,
                   device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
        seed_everything(args.seed)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, h, w = train(args, run, dev)
        cfg["train_seconds"] = round(time.time() - t0, 1)
        cfg["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None
        (run / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), run / "model.pt")
        print(f"学習 {cfg['train_seconds']} 秒, VRAM最大 {cfg['peak_vram_mb']} MB")

    for jp in JITTERS:
        title = f"step1 λ={cfg['lam']:g} seed{cfg['seed']} ずれ{jp:g}"
        s = evaluate(model, h, w, dev, run / f"eval_j{jp:g}", title, jp, args.eval_seed, pictures=(jp == 0))
        print(f"ずれ{jp:<4g}:", brief(s))
    print("出力:", run)


if __name__ == "__main__":
    main()
