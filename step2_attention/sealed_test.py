"""
Step 2 の封印テスト（DESIGN.md 17章。**一度だけ回す**。手順は 2026-09-30 に走らせる前に決めてコミットした）。

    .venv/Scripts/python.exe step2_attention/sealed_test.py            # 本番（一度だけ）
    .venv/Scripts/python.exe step2_attention/sealed_test.py --smoke 1  # 試運転（封印の材料に触れない）

- レシピ（凍結、git のタグ step2-recipe-frozen）：DNA なしの B（4歩先の驚き、元の引き金、θ=4、DNA なしの書き換え、標準の世界）。学習はない
- 4本＝評価の世界の乱数 9101〜9104。世界は S0〜S3 で判定、S4 は記録だけ（sealed.py）。σ の下限は0.02
- 大人の時期の組は封印の3つの変化（11→19、26→8、14→40。どれも4歩先に見えるので、「発火しない」の確かめはない）。育つ時期はこの3つの P₁ から
- θ の格子 {2,3,4,6,8}（評価だけ）
- 対照：時計 k ∈ {2,4,10,30,100,300}、サイコロ p ∈ {0.003,…,0.5}。書き換え方は各点で {DNA なし、その点の発火のしかたで学んだ DNA（封印の学習シード9001、
  runs/sealed_ctrl_*_s9001）} の良いほう（7.5 の決まり）。学んだ DNA が崩れていたら（学習の更新を10%より多く飛ばした、重みに nan、学習の終わりの NLL が始めより悪い）、
  その点は DNA なしだけにして記録する
- 判定 (i)：それぞれの世界・評価の乱数で、育つ時期と大人の時期の両方について、θ の格子のどの点でも時計とサイコロに負けない（上回るか決められない）、
  かつ θ=4 で両方を上回る。世界ごとに4本中3本以上なら合格。S0〜S3 のすべてが合格なら、封印テストは合格
- 参考（判定には使わない。先に回す）：標準の世界・評価の乱数1000・開発の組（24→36 は確かめ）で、同じ対照と比べる。対照の DNA が壊れていないかを見るため
- 試運転：封印でない世界（格子36、ノイズ0.022）、開発の評価の乱数1001、開発のシードで数回だけ学んだ対照の DNA（runs/smoke_ctrl_*_s0）。封印のシード・周期・世界は使わない
出力：runs/sealed_test/（summary.md、summary.json）。試運転は runs/smoke_sealed_test/
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import world as W  # noqa: E402
from model import AttentionCell, Physio2  # noqa: E402
from protocol import PAIRS_ORIGINAL, compare, envelope, run_period  # noqa: E402
from sealed import SEALED_CHANGES, SEALED_EVAL_SEEDS, SEALED_SIGMA_MIN, SEALED_WORLDS  # noqa: E402

THETAS = (2.0, 3.0, 4.0, 6.0, 8.0)
HEADLINE = 4.0
CLOCK_KS = (2, 4, 10, 30, 100, 300)
DICE_PS = (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)
JUDGED = ("S0", "S1", "S2", "S3")
PJ = {"grow": "育つ時期", "adult": "大人の時期"}


def controls():
    return ([("時計", {"kind": "clock", "k": k}, f"clock_k{k}") for k in CLOCK_KS] +
            [("サイコロ", {"kind": "dice", "p": p}, f"dice_p{p:g}") for p in DICE_PS])


def ctrl_ok(run):
    """対照の DNA が崩れていないか（17章）：更新を飛ばしたのが10%以下、重みがすべて有限、学習の終わりの NLL が始めより悪くない"""
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    sd = torch.load(run / "model.pt", map_location="cpu")
    finite = all(torch.isfinite(v).all() for v in sd.values())
    rows = list(csv.DictReader(open(run / "train_log.csv", encoding="utf-8")))
    nl = [float(r["nll_wave"]) for r in rows]
    head, tail = sum(nl[:50]) / max(len(nl[:50]), 1), sum(nl[-50:]) / max(len(nl[-50:]), 1)
    skipped = cfg.get("skipped_updates", 0) or 0
    ok = finite and skipped <= 0.1 * cfg["iters"] and math.isfinite(tail) and tail <= head
    return ok, dict(skipped=skipped, finite=finite, nll_head=head, nll_tail=tail)


def evaluate(wcfg, pairs, eval_seed, dev, ctrl_models, quiet):
    """1つの世界・評価の乱数で、B（DNA なし）の θ の格子と対照の曲線を出し、(i) を判定する"""
    dnaless = AttentionCell(dna=False).to(dev).eval()
    rp = lambda cell, pol, period: run_period(cell, pol, wcfg, period, eval_seed=eval_seed, device=dev, pairs=pairs, quiet=quiet)
    r = dict(theta={}, controls=[])
    for th in THETAS:
        old = Physio2.theta
        Physio2.theta = th
        try:
            for period in ("grow", "adult"):
                q = rp(dnaless, {"kind": "trigger"}, period)
                r["theta"][f"{period}_{th:g}"] = dict(fires=q["fires"], quality=q["quality"],
                                                       quiet_check=q.get("quiet_check"))
        finally:
            Physio2.theta = old
    for fam, pol, name in controls():
        writers = [("DNA なし", dnaless)] + ([("学んだ DNA", ctrl_models[name])] if ctrl_models.get(name) is not None else [])
        for wname, cell in writers:
            for period in ("grow", "adult"):
                q = rp(cell, pol, period)
                r["controls"].append(dict(family=fam, name=name, writer=wname, period=period, fires=q["fires"], quality=q["quality"]))
    verd, ok = {}, True
    for period in ("grow", "adult"):
        envs = {fam: envelope([(p["fires"], p["quality"]) for p in r["controls"] if p["period"] == period and p["family"] == fam])
                for fam in ("時計", "サイコロ")}
        for th in THETAS:
            b = r["theta"][f"{period}_{th:g}"]
            for fam, env in envs.items():
                v, qc = compare((b["fires"], b["quality"]), env)
                verd[f"{period}_{th:g}_{fam}"] = dict(verdict=v, fires=b["fires"], nll=b["quality"], control_nll=qc)
                if v == "負け" or (th == HEADLINE and v != "上回る"):
                    ok = False
    r["verdicts"], r["passed"] = verd, ok
    return r


def table(r):
    L = ["| 時期 | θ | 発火数 | NLL | 時計（同じ発火数） | サイコロ（同じ発火数） |", "|---|---:|---:|---:|---|---|"]
    for period in ("grow", "adult"):
        for th in THETAS:
            c1, c2 = r["verdicts"][f"{period}_{th:g}_時計"], r["verdicts"][f"{period}_{th:g}_サイコロ"]
            mk = "**" if th == HEADLINE else ""
            L.append(f"| {PJ[period]} | {mk}{th:g}{mk} | {c1['fires']:.2f} | {c1['nll']:.3f} | {c1['verdict']}（{c1['control_nll']:.3f}） | "
                     f"{c2['verdict']}（{c2['control_nll']:.3f}） |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert Physio2.surprise_ahead == 4 and W.WorldConfig().noise == 0.02, "凍結したレシピ（標準の世界）で回す"
    Physio2.sigma_min = SEALED_SIGMA_MIN
    t0 = time.time()
    if args.smoke:                                            # 封印の材料に触れない試運転
        worlds = {"試運転の世界": dict(size=36, zone=28, v=0.25, noise=0.022)}
        seeds = (1001,)
        pairs_sealed = ((12.0, 20.0), (24.0, 12.0))
        ctrl_dir, suffix = "smoke_ctrl_{}_s0", "_s0"
        out = HERE / "runs" / "smoke_sealed_test"
    else:
        worlds = SEALED_WORLDS
        seeds = SEALED_EVAL_SEEDS
        pairs_sealed = SEALED_CHANGES
        ctrl_dir, suffix = "sealed_ctrl_{}_s9001", "_s9001"
        out = HERE / "runs" / "sealed_test"
        if (out / "summary.json").exists():
            raise SystemExit("封印テストはもう回した（一度だけ。DESIGN.md 17章）")
    out.mkdir(parents=True, exist_ok=True)
    ctrl_models, ctrl_state = {}, {}
    for fam, pol, name in controls():
        run = HERE / "runs" / ctrl_dir.format(name)
        if not (run / "model.pt").exists():
            if args.smoke:
                ctrl_models[name] = None
                ctrl_state[name] = dict(ok=False, why="試運転では学んでいない点")
                continue
            raise SystemExit(f"対照の DNA がない：{run}")
        ok, info = ctrl_ok(run)
        ctrl_state[name] = dict(ok=ok, **info)
        if ok:
            m = AttentionCell(dna=True).to(dev).eval()
            m.load_state_dict(torch.load(run / "model.pt", map_location=dev))
            ctrl_models[name] = m
        else:
            ctrl_models[name] = None
    res = dict(smoke=bool(args.smoke), recipe="DNA なしの B（git のタグ step2-recipe-frozen）", controls=ctrl_state, reference=None, worlds={})
    # 参考（判定には使わない）：標準の世界・評価の乱数1000・開発の組
    ref = evaluate(W.WorldConfig(), PAIRS_ORIGINAL, 1000, dev, ctrl_models, quiet=((24.0, 36.0),))
    res["reference"] = ref
    print(f"参考（標準の世界、評価の乱数1000）：{'満たす' if ref['passed'] else '満たさない'}", flush=True)
    for wname, wc in worlds.items():
        wcfg = W.WorldConfig(**wc)
        per = {}
        for seed in seeds:
            per[str(seed)] = evaluate(wcfg, pairs_sealed, seed, dev, ctrl_models, quiet=())
            print(f"{wname} 評価の乱数{seed}：{'満たす' if per[str(seed)]['passed'] else '満たさない'}", flush=True)
        n_ok = sum(r["passed"] for r in per.values())
        res["worlds"][wname] = dict(config=wc, per_seed=per, n_ok=n_ok, passed=n_ok >= 3 if not args.smoke else None)
    judged = [w for w in JUDGED if w in res["worlds"]]
    res["passed"] = all(res["worlds"][w]["passed"] for w in judged) if judged and not args.smoke else None
    res["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# Step 2 の封印テスト（DESIGN.md 17章。一度だけ）" + ("（試運転。封印の材料は使っていない）" if args.smoke else ""), "",
         "レシピ：DNA なしの B（凍結、git のタグ `step2-recipe-frozen`）。判定は卒業の条件 (i)（θ の格子のどの点でも負けない、かつ θ=4 で上回る。両方の時期）。", ""]
    if not args.smoke:
        L += [f"**封印テストの判定：{'合格（Step 2 は卒業）' if res['passed'] else '不合格（(i) を満たさずに Step 2 を閉じる）'}**", ""]
    L += ["| 世界 | 満たした本数（4本中） | 判定 |", "|---|---:|---|"]
    for wname, w in res["worlds"].items():
        judge = "記録だけ" if wname == "S4" else ("—" if args.smoke else ("合格" if w["passed"] else "不合格"))
        L.append(f"| {wname}（{w['config']}） | {w['n_ok']} / {len(w['per_seed'])} | {judge} |")
    L += ["", "## 対照の DNA", "", "| 対照の点 | 使った | 更新を飛ばした | 学習の NLL（始め → 終わり） |", "|---|---|---:|---|"]
    for name, st in ctrl_state.items():
        L.append(f"| {name} | {'学んだ DNA と DNA なしの良いほう' if st.get('ok') else 'DNA なしだけ'} | {st.get('skipped', '—')} | "
                 + (f"{st['nll_head']:.3f} → {st['nll_tail']:.3f}" if 'nll_head' in st else st.get('why', '—')) + " |")
    L += ["", "## 参考（判定には使わない）：標準の世界・評価の乱数1000・開発の組", ""] + table(ref) + [""]
    for wname, w in res["worlds"].items():
        for seed, r in w["per_seed"].items():
            L += [f"## {wname}・評価の乱数{seed}：{'満たす' if r['passed'] else '満たさない'}", ""] + table(r) + [""]
    L += [f"かかった時間 {res['seconds']:.0f}秒", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]), flush=True)


if __name__ == "__main__":
    main()
