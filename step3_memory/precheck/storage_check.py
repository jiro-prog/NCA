"""
しまうタイミングの診断（step3_memory/DESIGN.md 12.3。読み方は走らせる前に書いた。記録だけで、判定はしない）。

    .venv/Scripts/python.exe step3_memory/precheck/storage_check.py [--device cuda]

眠る段（cell.Cell3(sleep=True)）を開発の世界で回し、そのそばで保存先の帳簿だけをつける（思い出しはしないので、セルの動きは眠る段と1ビットも変わらない）。
しまうタイミング3つを同じ回で帳簿づけする：
  (し1) 静かな時期の2回目以降の見直しでしまう（最後の驚きの発火から300・700・1500…ステップ目）
  (し2) 毎回の見直しでしまう（100ステップ目から）
  (し3) 離れるときにしまう（100ステップ以上静かだったあとの最初の驚きの発火で、それまでの係数を、0に戻す前の長い痕跡で比べる）
しまい方（12.2）：見直しの材料の長い痕跡 R_L の上で、いちばん合う保存先 u との当てはまりの差 (u − w)ᵀR_L(u − w) ≤ 6σ² なら u ← u ＋ (w − u)/(n＋1)（寄せる）、
なければ空いている保存先へ、いっぱいなら、いちばん長く使っていない保存先を入れ替える（12.4）。K=4。
測るもの（条件ごと、波の区域）：最後の保存先の使用数の分布、いっぱいになったことのあるセルの数、入れ替えの回数、どのパターンにも当たらない保存先の数、
同じパターンに2つ以上当たっている（重複の）保存先の数。当たる＝そのパターンの正確な係数との、きれいな波の上での1歩先の予測の差（二乗平均の平方根）≤ 0.02。
出力：step3_memory/runs/precheck_storage/summary.md、summary.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "step3_memory"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cell as C  # noqa: E402
import evaluate as E  # noqa: E402
from evaluate import W, Physio2  # noqa: E402

K = 4
TAU = 6.0
MATCH = 0.02                       # 当たる：きれいな波の上での1歩先の予測の差がセンサーのノイズ以下
FULL_MAX = 76                      # 読み方：いっぱいになったことのあるセルが、どの条件でも76個（波の区域7680 の1%）以下
VARIANTS = ("し1", "し2", "し3")


def quad(R6, u):
    """uᵀRu（R は上三角の6つ：00,01,02,11,12,22）。u は (B,3,H,W) → (B,H,W)"""
    a, b, c, d, e, f = R6.unbind(1)
    u0, u1, u2 = u.unbind(1)
    return u0 * u0 * a + 2 * u0 * u1 * b + 2 * u0 * u2 * c + u1 * u1 * d + 2 * u1 * u2 * e + u2 * u2 * f


class Book:
    """1つのしまうタイミングの、保存先の帳簿（セルごと K 個）"""

    def __init__(self, B, H, Wd, dev):
        self.slot = torch.zeros(B, K, 3, H, Wd, device=dev)
        self.n = torch.zeros(B, K, H, Wd, device=dev)              # 何回しまったか（0 は空き）
        self.age = torch.zeros(B, K, H, Wd, device=dev)            # 最後に使ってからのステップ数
        self.evict = torch.zeros(B, H, Wd, device=dev)
        self.new = torch.zeros(B, H, Wd, device=dev)
        self.merge = torch.zeros(B, H, Wd, device=dev)
        self.ever_full = torch.zeros(B, H, Wd, dtype=torch.bool, device=dev)

    def tick(self):
        self.age = self.age + (self.n > 0).float()

    def store(self, M, w, RL, sig2):
        """M (B,H,W) のセルに係数 w (B,3,H,W) をしまう（R_L は比べる材料、sig2 は σ²）"""
        if not bool(M.any()):
            return
        occ = self.n > 0
        d = torch.stack([quad(RL, self.slot[:, i] - w) for i in range(K)], 1)            # (B,K,H,W)
        d = torch.where(occ, d, torch.full_like(d, float("inf")))
        d_best, i_best = d.min(1)
        merge = M & (d_best <= TAU * sig2)
        newm = M & ~merge
        # 寄せる
        onehot = torch.nn.functional.one_hot(i_best, K).permute(0, 3, 1, 2).bool() & merge.unsqueeze(1)   # (B,K,H,W)
        n1 = self.n + 1
        upd = self.slot + (w.unsqueeze(1) - self.slot) / n1.unsqueeze(2)
        self.slot = torch.where(onehot.unsqueeze(2), upd, self.slot)
        self.n = torch.where(onehot, n1, self.n)
        self.age = torch.where(onehot, torch.zeros_like(self.age), self.age)
        self.merge += merge.float()
        # 新しくしまう：空きがあればいちばん若い番号の空き、なければいちばん長く使っていないもの
        empty = ~occ
        has_empty = empty.any(1)
        ar = torch.arange(K, device=w.device).view(1, K, 1, 1).expand(self.n.shape)
        idx_empty = torch.where(empty, ar, torch.full_like(ar, K)).min(1).values
        idx_lru = torch.where(occ, self.age, torch.full_like(self.age, -1.0)).max(1).indices
        idx = torch.where(has_empty, idx_empty, idx_lru)
        onehot2 = torch.nn.functional.one_hot(idx.clamp(max=K - 1), K).permute(0, 3, 1, 2).bool() & newm.unsqueeze(1)
        self.slot = torch.where(onehot2.unsqueeze(2), w.unsqueeze(1).expand_as(self.slot), self.slot)
        self.n = torch.where(onehot2, torch.ones_like(self.n), self.n)
        self.age = torch.where(onehot2, torch.zeros_like(self.age), self.age)
        self.new += (newm & has_empty).float()
        self.evict += (newm & ~has_empty).float()
        self.ever_full |= (self.n > 0).all(1)


def pred_gap(u, P):
    """きれいな波（平均0.5・振幅0.4・周期 P）の上で、係数 u と正確な係数の1歩先の予測の差の二乗平均の平方根。u は (...,3)"""
    c = math.cos(2 * math.pi / P)
    dw1, dw2, db = u[..., 0] - 2 * c, u[..., 1] + 1.0, u[..., 2] - (1 - c)
    m, a = 0.5, 0.4
    return torch.sqrt((dw1 * m + dw2 * m + db) ** 2 + a * a / 2 * (dw1 ** 2 + dw2 ** 2 + 2 * dw1 * dw2 * c))


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
    books = {v: Book(B, S_, S_, dev) for v in VARIANTS}
    for t in range(T):
        j = t // plan.dwell
        if t % plan.dwell == plan.dwell - 1 and j + 1 < len(plan.seq):
            ws = W.set_period(ws, sched[j + 1].to(dev))
        sense = W.sense(ws, wcfg, g)
        prev = st
        st, rec = cell.step(st, sense)
        for b in books.values():
            b.tick()
        sl = rec["sleep"][:, 0]
        sig2 = st.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0]
        books["し1"].store(sl & (st.k[:, 0] >= 2), st.core.w, st.RL, sig2)
        books["し2"].store(sl & (st.k[:, 0] >= 1), st.core.w, st.RL, sig2)
        leave = rec["surprise"][:, 0] & (prev.q[:, 0] >= C.Physio3.n_q)
        books["し3"].store(leave, prev.core.w, prev.RL, prev.core.v.clamp(min=Physio2.sigma_min ** 2)[:, 0])
        ws = W.advance(ws)
    wave = ~W.zone_mask(wcfg, dev)
    out = {}
    for v, b in books.items():
        rows = []
        for gi, grp in enumerate(plan.groups):
            sl_ = slice(gi * plan.n_per, (gi + 1) * plan.n_per)
            n = b.n[sl_][:, :, wave]                                                     # (n_per, K, セル)
            occ = n > 0
            used = occ.sum(1)                                                            # (n_per, セル)
            slots = b.slot[sl_].permute(0, 1, 3, 4, 2)[:, :, wave]                        # (n_per, K, セル, 3)
            gaps = torch.stack([pred_gap(slots, P) for P in grp], -1)                     # (n_per, K, セル, 3パターン)
            best_gap, best_p = gaps.min(-1)
            hit = occ & (best_gap <= MATCH)
            junk = occ & ~hit
            dup = 0
            for p in range(3):
                k_p = (hit & (best_p == p)).sum(1)                                       # そのパターンに当たる保存先の数
                dup += int((k_p - 1).clamp(min=0).sum())
            rows.append(dict(group="/".join(f"{p:g}" for p in grp),
                             used_hist=[int((used == u).sum()) for u in range(K + 1)],
                             ever_full=int(b.ever_full[sl_][:, wave].sum()), evictions=int(b.evict[sl_][:, wave].sum()),
                             new=int(b.new[sl_][:, wave].sum()), merges=int(b.merge[sl_][:, wave].sum()),
                             junk=int(junk.sum()), dup=dup, occupied=int(occ.sum()), n_cells=int(used.numel())))
        out[v] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    wcfg = W.WorldConfig(noise=0.02)
    res = {}
    for start in ("adult", "grow"):
        for seed in (300, 301):
            res[f"{start}|{seed}"] = run(wcfg, seed, start, args.device)
            print(start, seed, "done", flush=True)
    L = ["# しまうタイミングの診断（DESIGN.md 12.3。記録だけ。読み方は走らせる前に決めた）", "",
         "K=4、いっぱいならいちばん長く使っていない保存先を入れ替える。波の区域のセルは条件ごとに7680。使用数の分布は、最後に保存先を0・1・2・3・4個使っているセルの数。",
         "当たらない＝どのパターンの正確な係数とも、きれいな波の上での1歩先の予測の差が0.02 を超える保存先。重複＝同じパターンに2つ目以降が当たっている保存先", "",
         "| しまうタイミング | 始め方 | 評価の乱数 | 組 | 使用数の分布（0/1/2/3/4） | いっぱいになったことのあるセル | 入れ替え | 新しく | 寄せた | 当たらない | 重複 |",
         "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    worst = {v: 0 for v in VARIANTS}
    for v in VARIANTS:
        for key, out in res.items():
            start, seed = key.split("|")
            for r in out[v]:
                worst[v] = max(worst[v], r["ever_full"])
                L.append(f"| {v} | {'大人' if start == 'adult' else '育ち'} | {seed} | {r['group']} | {'/'.join(map(str, r['used_hist']))} | {r['ever_full']} | "
                         f"{r['evictions']} | {r['new']} | {r['merges']} | {r['junk']} | {r['dup']} |")
    v1 = worst["し1"]
    L += ["", "## 読み方（12.3）", "",
          f"- (し1) のいっぱいになったことのあるセルの最大：{v1}（基準 ≤ {FULL_MAX}）→ **{'この世界では無駄に埋まらない' if v1 <= FULL_MAX else '超える条件がある → 案を決め直す材料としてそうに出す'}**",
          f"- 記録：(し2) の最大 {worst['し2']}、(し3) の最大 {worst['し3']}"]
    out_dir = ROOT / "step3_memory" / "runs" / "precheck_storage"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
