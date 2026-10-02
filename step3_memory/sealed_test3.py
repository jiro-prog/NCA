"""
Step 3 の封印テスト（一度だけ）。step3_memory/DESIGN.md 10.1〜10.3（2026-10-01・02 に、そうと決めた）。

    .venv/Scripts/python.exe step3_memory/sealed_test3.py            # 本番（一度だけ。runs/sealed_test3/summary.json があれば回らない）
    .venv/Scripts/python.exe step3_memory/sealed_test3.py --dry 1    # 試運転（封印の材料に触れない。書き込みの数だけを出す）

問い：繰り返し切り替わる世界で、眠りと記憶によって学び続けられるか
  主張1（睡眠）：眠る段は、床より切り替えを見逃さない
  主張2（記憶）：記憶の段は、眠る段より、少ない書き込みで、または同じ書き込みでより良く、前に見たパターンに戻れる。初めてのパターンでは悪くならない
回すもの（どの世界も、どの本も）：床 Cell3(sleep=False)、眠る段 Cell3(sleep=True)、記憶の段 Cell3(sleep=True, memory=True)、天井＋睡眠（記録だけ）
判定（10.2・10.3。1本＝評価の乱数1つ。始め方は育ち、判定する切り替えは7つすべて）：
  主張1：1本の中のどの組でも、眠る段の見逃しの数×10 ≤ 床の見逃しの数、かつ眠る段の書き込み（驚き＋睡眠、全セルの1セル・1切り替えあたり）≤ 10
  主張2：1本の中のどの組でも、戻りの切り替えの平均で (A) 書き込み ≤ 眠る段 − 0.5 かつ窓のナット ≤ 眠る段＋2、または (B) 窓のナット ≤ 眠る段 − 2 かつ書き込み ≤ 眠る段。
         かつ守り：その本の初めての切り替え（2組×2回＝4回）を合わせて、ナットの差の合計 ≤ 2×4、書き込みの差の合計 ≤ 0.5×4
  世界ごとに4本中3本以上で満たせば、その世界は合格。判定する世界（T0〜T3）のすべてが合格なら、その主張は合格。主張1・主張2 は別々に出し、10.1 の閉じ方の表に当てはめる
記録（合否に使わない）：床の見逃しが判定するセル×切り替えの1割未満なら「慣れが起きにくい世界での合格」と注記、主張1 が落ちた理由（見逃し／費用）、
  (上2) の比（眠る段の睡眠の書き込み÷驚きの発火）、書き込みの内訳、眠る段と天井＋睡眠の差、記憶の段の取り違え（書き直し／害のあるもの）・見逃し・保存先の使用数・
  いっぱい・入れ替え・比べた保存先の数・何回目の発火で思い出したか、メモリ（1セルの状態の数）、T4 とタイミングのずれの結果
守り：本番は、凍結のタグ（step3-recipe-frozen）とレシピ・評価・このスクリプトのファイルが同じときだけ回る。結果は summary.json を先に書く（報告の組み立てで止まっても数字は残る）。
  回している間に出すのは、かかった時間だけ
試運転：封印でない世界（sealed3.DRY_*。開発の周期、開発の範囲の評価の乱数）で最後まで回し、報告の組み立てまで通す。出すのは、書き込みの数（眠る段と記憶の段の
  1切り替えあたり、全セル）と、最後まで動いたか、出力がそろったか、封印の材料に触れていないかだけ。ほかの数（ナット・見逃しなど）は計算するが、表示も保存もしない
  （2026-10-02、そうの判断）。書き込みが10回以下ならそのまま凍結へ、超えたら凍結の前にそうに相談
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cell as C  # noqa: E402
import evaluate as E  # noqa: E402
import sealed3 as S3  # noqa: E402
from evaluate import W  # noqa: E402

RUNGS = ("床", "眠る段", "記憶の段", "天井＋睡眠")
W_CAP = 10.0            # 書き込みの上限（(上3)、1切り替えあたり）
W_MARGIN = 0.5          # 書き込みの線
N_MARGIN = 2.0          # 窓のナットの線（引き分けの幅）
MISS_RATIO = 10         # 見逃しは床の10分の1 以下
FLOOR_EASY = 0.1        # 床の見逃しが判定するセル×切り替えの1割未満なら注記
N_PASS = 3              # 世界ごとに4本中3本
FROZEN_TAG = "step3-recipe-frozen"
FROZEN_PATHS = ("step3_memory/cell.py", "step3_memory/evaluate.py", "step3_memory/sealed3.py", "step3_memory/sealed_test3.py", "step2_attention/*.py")


def same_as(ref):
    """凍結するファイル（レシピ・評価・封印の定義・このスクリプト・Step 2 のコード）が ref と同じか（作業中の変更もないか）"""
    root = HERE.parent
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"{ref}^{{commit}}"], cwd=root, capture_output=True).returncode != 0:
        return False, f"{ref} がない"
    d = subprocess.run(["git", "diff", "--quiet", ref, "--", *FROZEN_PATHS], cwd=root, capture_output=True)
    return d.returncode == 0, (f"凍結するファイルは {ref} と同じ" if d.returncode == 0 else f"凍結するファイルが {ref} と違う")


def recipe_values():
    p2 = {k: v for k, v in vars(C.Physio2).items() if not k.startswith("_") and isinstance(v, (int, float))}
    return dict(Physio2=p2, Physio3=C.Physio3.to_dict())


def make_cell(rung, dev):
    """段の名前 → (セル, 天井か)"""
    if rung == "床":
        return C.Cell3(sleep=False, device=dev), False
    if rung == "眠る段":
        return C.Cell3(sleep=True, device=dev), False
    if rung == "記憶の段":
        return C.Cell3(sleep=True, memory=True, device=dev), False
    if rung == "天井＋睡眠":
        return C.Cell3(sleep=True, device=dev), True
    raise ValueError(rung)


def kinds(seq):
    """育ちから始めるので、判定する切り替えは全部。戻り＝前に滞在したパターンへ、初めて＝一度も滞在していないパターンへ"""
    return {sw: ("初めて" if seq[sw + 1] not in seq[:sw + 1] else "戻り") for sw in range(len(seq) - 1)}


def per_switch(res, plan, gi):
    """組 gi の切り替えごとの数字。書き込み（驚き＋睡眠）・内訳は全セルの1セルあたり、窓のナット・見逃しは波の区域（10.2）"""
    sl = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
    wave = res["wave"]
    rpc = wave.float().sum(0)
    rows = []
    for sw in range(len(plan.seq) - 1):
        q_col = (res["nll_sum"][sw][sl] * wave).sum((0, 1)) / (rpc * plan.n_per * plan.window)
        rows.append(dict(
            writes=float((res["fires_ph"][sw][sl] + res["sleeps_ph"][sw][sl]).mean()),
            surprise=float(res["fires_ph"][sw][sl].mean()), sleeps=float(res["sleeps_ph"][sw][sl].mean()),
            recalls=float(res["recalls_ph"][sw][sl].mean()),
            nats=float(q_col.mean()) * plan.window, miss=int((~res["react"][sw][sl][:, wave]).sum())))
    return rows


def judge_run(R, plan, wcfg):
    """1本（評価の乱数1つ）の主張1・主張2 と記録。R は段ごとの evaluate.run の結果"""
    kd = kinds(plan.seq)
    ret = [sw for sw, k in kd.items() if k == "戻り"]
    nov = [sw for sw, k in kd.items() if k == "初めて"]
    n_sw = len(plan.seq) - 1
    n_judged = int(R["床"]["wave"].sum()) * plan.n_per * n_sw          # 判定する（波の区域のセル×切り替え）の数（組ごと）
    out = dict(groups=[], claim1=True, claim2_returns=True, n_judged=n_judged,
               nan={k: bool(torch.isnan(R[k]["nll_sum"]).any()) for k in RUNGS},
               elapsed={k: R[k]["elapsed"] for k in RUNGS})
    g_dn, g_dw, g_n = 0.0, 0.0, 0
    for gi in range(len(plan.groups)):
        rows = {k: per_switch(R[k], plan, gi) for k in RUNGS}
        rs, rm, rc = rows["眠る段"], rows["記憶の段"], rows["天井＋睡眠"]
        miss = {k: sum(r["miss"] for r in rows[k]) for k in RUNGS}
        w_s = float(np.mean([r["writes"] for r in rs]))
        ok_miss = MISS_RATIO * miss["眠る段"] <= miss["床"]
        ok_cost = w_s <= W_CAP
        mean_ret = lambda rr, key: float(np.mean([rr[sw][key] for sw in ret]))  # noqa: E731
        ws_, wm, wc = mean_ret(rs, "writes"), mean_ret(rm, "writes"), mean_ret(rc, "writes")
        ns, nm, nc = mean_ret(rs, "nats"), mean_ret(rm, "nats"), mean_ret(rc, "nats")
        okA = bool((wm <= ws_ - W_MARGIN) and (nm <= ns + N_MARGIN))
        okB = bool((nm <= ns - N_MARGIN) and (wm <= ws_))
        for sw in nov:
            g_dn += rm[sw]["nats"] - rs[sw]["nats"]
            g_dw += rm[sw]["writes"] - rs[sw]["writes"]
            g_n += 1
        brk = {k: dict(surprise=float(np.mean([r["surprise"] for r in rows[k]])), sleeps=float(np.mean([r["sleeps"] for r in rows[k]])),
                       recalls=float(np.mean([r["recalls"] for r in rows[k]])), writes=float(np.mean([r["writes"] for r in rows[k]])))
               for k in RUNGS}
        sur_s, slp_s = brk["眠る段"]["surprise"], brk["眠る段"]["sleeps"]
        reason = "" if (ok_miss and ok_cost) else "・".join(x for x, bad in (("見逃し", not ok_miss), ("費用", not ok_cost)) if bad)
        out["groups"].append(dict(
            claim1=bool(ok_miss and ok_cost), fail1=reason, miss=miss, writes_sleep=w_s,
            floor_easy=bool(miss["床"] < FLOOR_EASY * n_judged), ratio_up2=(slp_s / sur_s if sur_s > 0 else None),
            returns_ok=bool(okA or okB), okA=okA, okB=okB,
            ret_writes=dict(sleep=ws_, memory=wm, ceiling=wc), ret_nats=dict(sleep=ns, memory=nm, ceiling=nc),
            breakdown=brk, rows=rows, summarize={k: E.summarize(R[k], wcfg, plan)[gi]["rows"] for k in RUNGS}))
        out["claim1"] &= bool(ok_miss and ok_cost)
        out["claim2_returns"] &= bool(okA or okB)
    out["guard"] = bool((g_dn <= N_MARGIN * g_n) and (g_dw <= W_MARGIN * g_n))
    out["guard_sums"] = dict(nats=g_dn, writes=g_dw, n_novel=g_n)
    out["claim2"] = bool(out["claim2_returns"] and out["guard"])
    mem = R["記憶の段"]
    wv = mem["wave"]
    nz = ~wv
    out["memory_watch"] = dict(
        mis_noop=int(mem["mis_noop"][:, wv].sum()), mis_change=int(mem["mis_change"][:, wv].sum()),
        mis_novel=int(mem["mis_novel"][:, wv].sum()), mis_return=int(mem["mis_return"][:, wv].sum()), rec_ok=int(mem["rec_ok"][:, wv].sum()),
        nth_hist=[int(x) for x in mem["nth_hist"]], comp=float(mem["comp"]),
        ever_full_wave=int(mem["ever_full"][:, wv].sum()), ever_full_noise=int(mem["ever_full"][:, nz].sum()),
        evict_wave=int(mem["evict"][:, wv].sum()), evict_noise=int(mem["evict"][:, nz].sum()),
        slot_used_wave=[int((mem["slot_used"][:, wv] == u).sum()) for u in range(C.Physio3.k_slots + 1)],
        slot_used_noise=[int((mem["slot_used"][:, nz] == u).sum()) for u in range(C.Physio3.k_slots + 1)])
    return out


def run_world(name, spec, groups, seeds, dev, sealed, jitter_p=0.0):
    wcfg = W.WorldConfig(size=spec["size"], zone=spec["zone"], v=spec["v"], noise=spec["noise"])
    plan = E.Plan(seq=S3.SEQ, groups=groups, n_per=8, dwell=spec["dwell"], window=300, end=(250, 300))
    judged = {}
    for seed in seeds:
        R = {}
        for rung in RUNGS:
            cell, ceil = make_cell(rung, dev)
            R[rung] = E.run(cell, wcfg, seed, "grow", plan=plan, dev=dev, ceiling=ceil, sealed=sealed, jitter_p=jitter_p)
            print(f"{name} 評価の乱数 {seed} {rung}: {R[rung]['elapsed']:.0f}秒", flush=True)
        judged[seed] = judge_run(R, plan, wcfg)
        del R
    return judged


def verdicts(results, judged_worlds):
    v = {}
    for claim in ("claim1", "claim2"):
        per_world = {n: dict(n_ok=sum(1 for j in results[n].values() if j[claim]), n=len(results[n])) for n in results}
        for n in per_world:
            per_world[n]["pass"] = per_world[n]["n_ok"] >= N_PASS
        v[claim] = dict(per_world=per_world, pass_=all(per_world[n]["pass"] for n in judged_worlds))
    return v


CLOSE = {
    (True, True): "卒業。凍結するレシピは3段目（睡眠＋記憶）",
    (True, False): "主張1 を満たし、主張2 を満たさずに閉じる。凍結するレシピは2段目（睡眠）。残す問い：記憶の価値（別枠の問い）",
    (False, True): "主張2 を満たし、主張1 を満たさずに閉じる。凍結するレシピは3段目。注意書き：睡眠の段は、見ていない世界で床より見逃しを減らせなかった。"
                   "記憶の効き目は、その眠る段との比べ。眠る段が慣れている世界では、記憶が正確な係数を書き戻すことで慣れも直るため、記憶の効き目に慣れの修復が混ざっている可能性がある。"
                   "残す問い：慣れ（睡眠の段の弱さ）。主張1 が費用だけで落ちたなら、残す問いは「睡眠の費用」",
    (False, False): "どちらも満たさずに閉じる。凍結しない（Step 2 の凍結したレシピのまま）。残す問い：慣れと記憶の両方（主張1 が費用だけで落ちたなら、慣れの代わりに睡眠の費用）",
}


def f2(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def run_table(world_name, judged, groups):
    """本ごと・組ごとの判定の数と記録の行"""
    L = []
    for seed, j in judged.items():
        for gi, g in enumerate(j["groups"]):
            grp = "/".join(f"{p:g}" for p in groups[gi])
            c1 = "満たす" if g["claim1"] else f"満たさない（{g['fail1']}）"
            if g["floor_easy"]:
                c1 += "、床の見逃しが1割未満"
            ab = ("A" if g["okA"] else "") + ("B" if g["okB"] else "")
            c2 = f"満たす（{ab}）" if g["returns_ok"] else "満たさない"
            m = g["miss"]
            rw, rn = g["ret_writes"], g["ret_nats"]
            b = g["breakdown"]["眠る段"]
            L.append(f"| {world_name} | {seed} | {grp} | {c1} | {m['床']}／{m['眠る段']}／{m['記憶の段']}／{m['天井＋睡眠']} | {g['writes_sleep']:.2f} | "
                     f"{b['sleeps']:.2f}÷{b['surprise']:.2f}＝{f2(g['ratio_up2'])} | {c2} | {rw['sleep']:.2f}→{rw['memory']:.2f}（天井 {rw['ceiling']:.2f}） | "
                     f"{rn['sleep']:.1f}→{rn['memory']:.1f}（天井 {rn['ceiling']:.1f}） |")
        gs = j["guard_sums"]
        L.append(f"| {world_name} | {seed} | 守り（初めて{gs['n_novel']}回） | | | | | {'満たす' if j['guard'] else '満たさない'}："
                 f"ナットの差の合計 {gs['nats']:+.1f}（≤ {N_MARGIN * gs['n_novel']:g}）、書き込みの差の合計 {gs['writes']:+.2f}（≤ {W_MARGIN * gs['n_novel']:g}） | | |")
    return L


TABLE_HEAD = ["| 世界 | 評価の乱数 | 組 | 主張1 | 見逃し 床／眠る段／記憶の段／天井＋睡眠 | 眠る段の書き込み | (上2) 睡眠÷驚き | 主張2 戻り | 戻りの書き込み 眠る段→記憶の段 | 戻りの窓のナット 眠る段→記憶の段 |",
              "|---|---:|---|---|---|---:|---|---|---|---|"]


def build_report(results, jitter_res, verdict, meta):
    """報告（summary.md の行と summary.json の中身）を組み立てる。試運転でも最後まで通す（表示も保存もしない）"""
    judged_worlds, groups, seeds, jit = meta["judged"], meta["groups"], meta["seeds"], meta["jitter"]
    c1, c2 = verdict["claim1"]["pass_"], verdict["claim2"]["pass_"]
    close = CLOSE[(c1, c2)]
    reasons = {"見逃し": 0, "費用": 0, "見逃し・費用": 0}
    for n in judged_worlds:
        for j in results[n].values():
            for g in j["groups"]:
                if g["fail1"]:
                    reasons[g["fail1"]] += 1
    easy = {n: sum(g["floor_easy"] for j in results[n].values() for g in j["groups"]) for n in results}
    n_rg = {n: sum(len(j["groups"]) for j in results[n].values()) for n in results}
    rec_worlds = [n for n in results if n not in judged_worlds]
    L = [f"# Step 3 の封印テスト（{meta['mode']}。DESIGN.md 10.1〜10.3）", "",
         f"世界 {', '.join(results)}（判定は {', '.join(judged_worlds)}、{', '.join(rec_worlds)} は記録だけ）＋{jit['world']} にタイミングのずれ p={jit['p']}（記録だけ）。"
         f"評価の乱数 {', '.join(map(str, seeds))}、組 {', '.join('/'.join(f'{p:g}' for p in g) for g in groups)}、並び {', '.join(S3.SEQ)}、育ちから。所要 {meta['elapsed'] / 60:.0f}分", "",
         "## 判定", "", "| 主張 | " + " | ".join(judged_worlds) + " | 合否 | " + " | ".join(f"{n}（記録）" for n in rec_worlds) + " |",
         "|---|" + "---:|" * len(judged_worlds) + "---|" + "---:|" * len(rec_worlds)]
    for claim, label in (("claim1", "主張1（睡眠）"), ("claim2", "主張2（記憶）")):
        pw = verdict[claim]["per_world"]
        L.append(f"| {label} | " + " | ".join(f"{pw[n]['n_ok']}/{pw[n]['n']}" for n in judged_worlds) +
                 f" | **{'合格' if verdict[claim]['pass_'] else '不合格'}** | " + " | ".join(f"{pw[n]['n_ok']}/{pw[n]['n']}" for n in rec_worlds) + " |")
    L += ["", f"- **閉じ方（10.1 の表）：{close}**",
          f"- 主張1 を満たさなかった本・組（判定する世界）：見逃しだけ {reasons['見逃し']}、費用だけ {reasons['費用']}、両方 {reasons['見逃し・費用']}",
          "- 床の見逃しが判定するセル×切り替えの1割未満だった本・組（「慣れが起きにくい世界」の注記）：" + "、".join(f"{n} {easy[n]}/{n_rg[n]}" for n in results),
          "- nan：" + ("なし" if not any(any(j["nan"].values()) for r in results.values() for j in r.values()) else "あり（JSON の nan を見る）"),
          f"- メモリ（1セルの状態の数）：床 {C.memory_numbers(sleep=False)}、眠る段 {C.memory_numbers(sleep=True)}、記憶の段 {C.memory_numbers(sleep=True, memory=True)}", "",
          "## 本ごと・組ごと（判定に使う数と記録。書き込みは全セルの1セル・1切り替えあたり、見逃しは波の区域のセル×切り替えの数、窓のナットは波の区域の1セル・1切り替えあたり）", ""]
    L += TABLE_HEAD
    for n in results:
        L += run_table(n, results[n], groups)
    L += ["", "## 書き込みの内訳（世界ごとに4本×2組の平均。全セルの1セル・1切り替えあたり。思い出しは驚きの発火の内訳）", "",
          "| 世界 | 段 | 驚きの発火 | うち思い出し | 睡眠の書き込み | 合計 | 睡眠の割合 |", "|---|---|---:|---:|---:|---:|---:|"]
    for n in results:
        for k in RUNGS:
            bs = [g["breakdown"][k] for j in results[n].values() for g in j["groups"]]
            s_, r_, z_, w_ = (float(np.mean([b[x] for b in bs])) for x in ("surprise", "recalls", "sleeps", "writes"))
            L.append(f"| {n} | {k} | {s_:.2f} | {r_:.2f} | {z_:.2f} | {w_:.2f} | {f2(z_ / w_ if w_ > 0 else None)} |")
    L += ["", "## 記憶の段の見張り（合否に使わない。数は本ごとに2組の合計）", "",
          "| 世界 | 評価の乱数 | 取り違え：書き直し／害のある | 初めての時期／戻りの時期 | 正しく思い出した | 何回目の発火か（1/2/3/4以上） | 驚きの発火1回あたりに比べた保存先の数 | "
          "保存先の使用数 0/1/2/3/4（波の区域） | 同じ（ノイズの区域） | いっぱいになったセル（波／ノイズ） | 入れ替え（波／ノイズ） |",
          "|---|---:|---|---|---:|---|---:|---|---|---|---|"]
    for n in results:
        for seed, j in results[n].items():
            m = j["memory_watch"]
            L.append(f"| {n} | {seed} | {m['mis_noop']}／{m['mis_change']} | {m['mis_novel']}／{m['mis_return']} | {m['rec_ok']} | {'/'.join(map(str, m['nth_hist']))} | "
                     f"{m['comp']:.2f} | {'/'.join(map(str, m['slot_used_wave']))} | {'/'.join(map(str, m['slot_used_noise']))} | "
                     f"{m['ever_full_wave']}／{m['ever_full_noise']} | {m['evict_wave']}／{m['evict_noise']} |")
    jn = f"{jit['world']}＋ずれ"
    L += ["", f"## タイミングのずれ（{jit['world']} に p={jit['p']}。記録だけ）", "",
          f"- 主張1 の形を満たした本：{sum(j['claim1'] for j in jitter_res.values())}/{len(jitter_res)}、主張2 の形：{sum(j['claim2'] for j in jitter_res.values())}/{len(jitter_res)}", ""]
    L += TABLE_HEAD
    L += run_table(jn, jitter_res, groups)
    js = dict(mode=meta["mode"], verdict=verdict, close=close, fail1_reasons=reasons, floor_easy=easy, elapsed=meta["elapsed"], recipe=recipe_values(),
              frozen=meta["frozen"], worlds=meta["worlds"], groups=groups, seeds=list(seeds), seq=list(S3.SEQ), jitter=jit,
              memory_numbers=dict(floor=C.memory_numbers(sleep=False), sleep=C.memory_numbers(sleep=True), memory=C.memory_numbers(sleep=True, memory=True)),
              results={n: {str(s): j for s, j in r.items()} for n, r in results.items()},
              jitter_results={str(s): j for s, j in jitter_res.items()})
    return L, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    P2, P3 = C.Physio2, C.Physio3
    assert (P2.surprise_ahead, P2.sigma_min, P2.theta) == (4, 0.02, 4.0), "凍結した床の値と違う"
    assert (P3.gamma_L, P3.n_q, P3.k_slots, P3.tau) == (0.99, 100, 4, 6.0), "凍結した睡眠・記憶の値と違う"
    if args.dry:
        worlds, judged_worlds, groups, seeds, jit, sealed = S3.DRY_WORLDS, S3.DRY_JUDGED, S3.DRY_GROUPS, S3.DRY_SEEDS, S3.DRY_JITTER, False
        assert all(s < 9000 for s in seeds) and not ({p for g in groups for p in g} & S3.SEALED_PERIODS), "試運転が封印の材料に触れている"
        out = HERE / "runs" / "dry_sealed_test3"
        frozen = same_as("HEAD")
        mode = "試運転"
    else:
        worlds, judged_worlds, groups, seeds, jit, sealed = S3.SEALED_WORLDS, S3.JUDGED, S3.SEALED_GROUPS, S3.SEALED_SEEDS, S3.JITTER, True
        out = HERE / "runs" / "sealed_test3"
        if (out / "summary.json").exists():
            raise SystemExit("封印テストはもう回した（runs/sealed_test3/summary.json がある）。二度目は回さない")
        frozen = same_as(FROZEN_TAG)
        if not frozen[0]:
            raise SystemExit(f"回さない：{frozen[1]}（凍結したレシピだけを封印テストにかける）")
        mode = "本番、一度だけ"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {name: run_world(name, spec, groups, seeds, dev, sealed) for name, spec in worlds.items()}
    jitter_res = run_world(f"{jit['world']}＋ずれ", worlds[jit["world"]], groups, seeds, dev, sealed, jitter_p=jit["p"])
    elapsed = time.time() - t0
    verdict = verdicts(results, judged_worlds)
    meta = dict(mode=mode, judged=judged_worlds, groups=[list(g) for g in groups], seeds=seeds, jitter=jit, elapsed=elapsed,
                worlds=worlds, frozen=dict(ref=("HEAD" if args.dry else FROZEN_TAG), same=frozen[0]))

    if args.dry:
        # 報告の組み立てまで通す（本番で止まらないことを確かめる）。中身は表示も保存もしない
        L, js = build_report(results, jitter_res, verdict, meta)
        md_ok, js_ok = len("\n".join(L)) > 0, len(json.dumps(js, ensure_ascii=False)) > 0
        wmax = {}
        for name, r in list(results.items()) + [(f"{jit['world']}＋ずれ", jitter_res)]:
            gs = [g for j in r.values() for g in j["groups"]]
            wmax[name] = (max(g["breakdown"]["眠る段"]["writes"] for g in gs), max(g["breakdown"]["記憶の段"]["writes"] for g in gs))
        w_all = max(max(v) for v in wmax.values())
        D = ["# Step 3 の封印テストの試運転（封印の材料に触れない。書き込みの数だけを出す）", "",
             f"- 世界 {', '.join(worlds)}（＋{jit['world']} にタイミングのずれ p={jit['p']}）、評価の乱数 {', '.join(map(str, seeds))}（開発の範囲）、"
             f"組 {', '.join('/'.join(f'{p:g}' for p in g) for g in groups)}（開発の周期）、育ちから",
             f"- 最後まで動いた：はい（{elapsed / 60:.0f}分。世界{len(worlds) + 1}×{len(seeds)}本×{len(RUNGS)}段＝{(len(worlds) + 1) * len(seeds) * len(RUNGS)}回）",
             f"- 出力がそろった：判定・記録・見張りの組み立て（summary.md の行 {'はい' if md_ok else 'いいえ'}、summary.json の中身 {'はい' if js_ok else 'いいえ'}）。中身は表示も保存もしていない",
             "- 封印の材料：使っていない（評価の乱数はすべて9000 未満、周期に封印の 13・48・7・30 なし、sealed=False で開発の守りを通った）",
             f"- 凍結の確かめのコード：{frozen[1]}（本番ではタグ {FROZEN_TAG} と比べる）", "",
             "| 世界 | 眠る段の書き込み（最大） | 記憶の段の書き込み（最大） |", "|---|---:|---:|"]
        D += [f"| {n} | {a:.2f} | {b:.2f} |" for n, (a, b) in wmax.items()]
        D += ["", f"書き込み＝驚きの発火＋睡眠の書き込み（思い出しは驚きの発火の内訳）。全セルの1セル・1切り替えあたりを組ごとに出し、{len(seeds)}本×{len(groups)}組の最大をとった。",
              "", f"- **書き込みの最大 {w_all:.2f} → {'10回以下：そのまま凍結へ' if w_all <= W_CAP else '10回を超えた：凍結の前にそうに相談'}**（2026-10-02 に先に決めた動き方）",
              "- ほかの数（ナット・見逃しなど）は計算したが、表示も保存もしていない（そうの判断）"]
        (out / "summary.md").write_text("\n".join(D) + "\n", encoding="utf-8")
        print("\n".join(D))
        return

    # 本番：数字（summary.json）を先に書き、そのあとで表（summary.md）
    L, js = build_report(results, jitter_res, verdict, meta)
    (out / "summary.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
