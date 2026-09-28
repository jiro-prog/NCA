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
    best/        良い重み（--val-every ごとに検証。BestKeeper を参照）：iterN/、front.json、history.csv
    checkpoints/ --save-every ごとの途中の重み
"""
import argparse
import json
import random
import shutil
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
    model = ConcentrationNCA(see_c=bool(args.see_c)).to(dev)
    model.use_checkpoint = args.checkpoint
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], gamma=0.1)
    pool = seed_state(h, w, POOL_SIZE)

    # 学習時のタイミングのずれ（環境のゆらぎ）。毎回、確率 p を 0〜train_jitter から選ぶ
    gen_p = torch.Generator().manual_seed(args.seed + 1)
    gen_mask = torch.Generator(device=dev)
    gen_mask.manual_seed(args.seed + 2)

    log = open(run / "train_log.csv", "w", encoding="utf-8", newline="")
    log.write("iter,steps,loss,rgba_loss,fire_cost,fires,alive_cell_steps,fire_rate,"
              "x_absmax,c_max,d_mean,jitter,birth_fires,hardness,seconds\n")
    if args.save_every:
        (run / "checkpoints").mkdir(exist_ok=True)
    keeper = BestKeeper(run, args.val_jitter, args.keep_max_loss, args.keep_min_heal) if args.val_every else None
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
        jp = float(torch.rand((), generator=gen_p)) * args.train_jitter if args.train_jitter > 0 else 0.0
        # 発火の固さ：anneal_iters 回かけて、なめらかな量（0）から 0/1（1）へ上げる。0 なら最初から 0/1
        hardness = min(1.0, i / args.anneal_iters) if args.anneal_iters > 0 else 1.0
        out, fire_sum, n_fire, n_alive, n_birth = model(s, steps, jitter_p=jp, generator=gen_mask,
                                                        hardness=hardness)
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
        with torch.no_grad():                   # 状態の大きさ（爆発の兆し）と、目覚ましの平均（眠り始めたか）
            live = model.alive(out.x)
            x_absmax = out.x.abs().max().item()
            c_max = out.c.max().item()
            d_mean = ((out.d * live).sum() / live.sum().clamp(min=1)).item()
        log.write(f"{i},{steps},{loss.item():.6g},{rgba_loss.item():.6g},{fire_cost.item():.6g},"
                  f"{nf},{na},{nf / max(na, 1):.6f},{x_absmax:.4g},{c_max:.4g},{d_mean:.4f},{jp:.3f},"
                  f"{int(n_birth.item())},{hardness:.3f},"
                  f"{time.time() - t0:.1f}\n")
        if args.save_every and (i + 1) % args.save_every == 0:
            torch.save(model.state_dict(), run / "checkpoints" / f"iter{i + 1}.pt")
        if keeper and (i + 1) % args.val_every == 0:
            rec, passed, on_front = keeper.offer(i + 1, model, h, w, dev)
            print(f"  検証 iter{i + 1}：成長後ロス {rec['after_loss']:.4f}  総発火 {rec['total_fires']:,}  "
                  f"維持期 {rec['maint_fps']:.1f}/ステップ  治った割合 {rec['heal']:.2f}  "
                  f"{'残す' if on_front else ('下限を満たす' if passed else '下限に届かず')}", flush=True)
        if i % 100 == 0 or i == args.iters - 1:
            log.flush()
            print(f"iter {i:5d}  rgba {rgba_loss.item():.5f}  発火率 {nf / max(na, 1):.3f}  "
                  f"({time.time() - t0:.0f}秒)", flush=True)
    log.close()
    if keeper:
        keeper.close()
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
        s, _, hard, pre, birth = self.model.step(s, self.jitter_p, self.gen)
        n = pre.sum().clamp(min=1)
        extra = dict(birth_fires=int(birth.sum().item()),
                     d_mean=float((s.d * pre).sum() / n), c_mean=float((s.c * pre).sum() / n))
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


class BestKeeper:
    """良い重みを残す。検証のロールアウト（3期プロトコル、ずれ jitter）で測り、
    質の下限（成長後の平均ロス ≤ max_loss、かつ、治った割合 ≥ min_heal）を満たすものの中から、
    「成長後の平均ロス」と「総発火」のどちらでも負けていないもの（パレートの前線）を全部残す。
    1つの重みづけの点数では選ばない（λ次第で「起きている良いモデル」か「眠っている良いモデル」の片方しか残らないため）。
    寝ているだけで傷を治さないモデル（治った割合≈0）や、形が崩れたモデルは、下限で落ちる。
    検証は全体の乱数を使わない（学習の道筋を変えない）。

    出力（run/best/）：iterN/（model.pt, summary.json, sheet.png, curves.png）、front.json、history.csv"""

    def __init__(self, run, jitter, max_loss, min_heal, gen_seed=1000):
        self.dir = run / "best"
        self.dir.mkdir(exist_ok=True)
        self.jitter, self.max_loss, self.min_heal, self.gen_seed = jitter, max_loss, min_heal, gen_seed
        self.front = []
        self.hist = open(self.dir / "history.csv", "w", encoding="utf-8", newline="")
        self.hist.write("iter,after_loss,total_fires,maint_fires_per_step,heal_fraction,passed,on_front\n")

    @staticmethod
    def _dominates(a, b):
        return (a["after_loss"] <= b["after_loss"] and a["total_fires"] <= b["total_fires"]
                and (a["after_loss"] < b["after_loss"] or a["total_fires"] < b["total_fires"]))

    @torch.no_grad()
    def offer(self, it, model, h, w, dev):
        proto = Protocol()
        rows, frames = run_protocol(Step1Adapter(model, h, w, dev, self.jitter, self.gen_seed),
                                    make_target().to(dev), proto)
        s = summarize(rows, proto)
        s["jitter_p"] = self.jitter
        rec = dict(iter=it, after_loss=s["after_loss"], total_fires=s["total_fires"],
                   maint_fps=s["maintain"]["fires_per_step"], heal=s["heal_fraction"])
        passed = rec["after_loss"] <= self.max_loss and rec["heal"] >= self.min_heal
        on_front = False
        if passed:
            cand = self.front + [rec]
            front = [a for a in cand if not any(self._dominates(b, a) for b in cand if b is not a)]
            for a in self.front:                    # 前線から落ちた重みは消す
                if a not in front:
                    shutil.rmtree(self.dir / f"iter{a['iter']}", ignore_errors=True)
            if rec in front:
                on_front = True
                d = self.dir / f"iter{it}"
                d.mkdir(exist_ok=True)
                torch.save(model.state_dict(), d / "model.pt")
                (d / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
                title = f"残した重み iter{it}（ずれ{self.jitter:g}）"
                viz.save_sheet(frames, SHEET_STEPS, d / "sheet.png", title=title)
                viz.save_curves(rows, proto, d / "curves.png", title=title)
            self.front = front
            (self.dir / "front.json").write_text(
                json.dumps(sorted(front, key=lambda a: a["total_fires"]), ensure_ascii=False, indent=2),
                encoding="utf-8")
        self.hist.write(f"{it},{rec['after_loss']:.6g},{rec['total_fires']},{rec['maint_fps']:.2f},"
                        f"{rec['heal']:.3f},{int(passed)},{int(on_front)}\n")
        self.hist.flush()
        return rec, passed, on_front

    def close(self):
        self.hist.close()


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
    ap.add_argument("--save-every", type=int, default=0, help="N回ごとに途中の重みを checkpoints/ に保存する")
    ap.add_argument("--see-c", type=int, default=1, help="1 なら濃度もルールのネットの入力に入れる（既定）。0 なら入れない")
    ap.add_argument("--coupling", choices=["sum", "mean", "cap"], default=None,
                    help="近所の変化イベントのまとめ方（合計／平均／合計に上限）。既定は Physio の値（sum）")
    ap.add_argument("--coupling-cap", type=float, default=None, help="--coupling cap のときの、合計の上限")
    ap.add_argument("--val-every", type=int, default=250, help="N回ごとに検証して、良い重みを best/ に残す（0 で使わない）")
    ap.add_argument("--val-jitter", type=float, default=0.1, help="検証のときのタイミングのずれ")
    ap.add_argument("--keep-max-loss", type=float, default=0.04, help="残す重みの下限：成長後の平均ロスがこれ以下")
    ap.add_argument("--keep-min-heal", type=float, default=0.2, help="残す重みの下限：治った割合がこれ以上")
    ap.add_argument("--anneal-iters", type=int, default=0,
                    help="学習の初期に発火をなめらかな量として扱い、この回数かけて 0/1 に近づける（0 なら最初から 0/1）")
    ap.add_argument("--d-bias", type=float, default=None,
                    help="目覚ましの初期値のずらし。目覚ましは sigmoid(ネットの出力＋d_bias)。既定は Physio の値（2.0）")
    ap.add_argument("--train-jitter", type=float, default=0.0,
                    help="学習時のタイミングのずれ。毎回、発火が1ステップ遅れる確率を 0〜この値 から選ぶ")
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-only", default=None)
    ap.add_argument("--ckpt", default=None, help="--eval-only と一緒に使う。checkpoints/ の途中の重みを評価する（例 iter1300）")
    ap.add_argument("--eval-seed", type=int, default=1000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.eval_only:
        run = Path(args.eval_only)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        for k, v in cfg.get("physio", {}).items():      # 学習したときの固定の定数（d_bias など）に戻す
            setattr(Physio, k, v)
        model = ConcentrationNCA(see_c=cfg.get("see_c", True)).to(dev)     # 古い run は濃度を入力に入れていた
        weights = run / "checkpoints" / f"{args.ckpt}.pt" if args.ckpt else run / "model.pt"
        model.load_state_dict(torch.load(weights, map_location=dev))
        h, w = make_target().shape[-2:]
    else:
        run = HERE / "runs" / (args.name or f"seed{args.seed}_lam{args.lam:g}")
        run.mkdir(parents=True, exist_ok=True)
        if args.d_bias is not None:
            Physio.d_bias = args.d_bias
        if args.coupling is not None:
            Physio.coupling = args.coupling
        if args.coupling_cap is not None:
            Physio.coupling_cap = args.coupling_cap
        if Physio.coupling == "cap" and Physio.coupling_cap is None:
            ap.error("--coupling cap には --coupling-cap が要る")
        cfg = dict(seed=args.seed, iters=args.iters, lam=args.lam, checkpoint=bool(args.checkpoint),
                   save_every=args.save_every, train_jitter=args.train_jitter, see_c=bool(args.see_c),
                   anneal_iters=args.anneal_iters, val_every=args.val_every, val_jitter=args.val_jitter,
                   keep_max_loss=args.keep_max_loss, keep_min_heal=args.keep_min_heal,
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

    tag = f"_{args.ckpt}" if args.ckpt else ""
    for jp in JITTERS:
        title = f"step1 λ={cfg['lam']:g} seed{cfg['seed']}{tag} ずれ{jp:g}"
        s = evaluate(model, h, w, dev, run / f"eval{tag}_j{jp:g}", title, jp, args.eval_seed)
        print(f"ずれ{jp:<4g}:", brief(s))
    print("出力:", run)


if __name__ == "__main__":
    main()
