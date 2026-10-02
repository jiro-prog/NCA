"""
Step 3 の封印テストの定義（step3_memory/DESIGN.md 10.3。2026-10-01・02 に、そうと決めた）。封印テストのスクリプト（sealed_test3.py）の本番の中でだけ使う。
開発のコードは、封印の乱数（9000以上）を拒む（step2_attention/sealed.check_dev_seed を evaluate.run が呼ぶ。本番だけ sealed=True で通す）。

- 4本＝評価の世界の乱数 9301〜9304（DNA なしなので学習のシードはない）。始め方は育ちだけ（予測器0から）
- 組（周期 A/B/C）は sealed_select3.py で式だけで選んだ（開発の 12/24/8・12/20/8 と同じくらいの見えにくさ。C は A・B から離す。性能は見ていない）
- 並びは開発と同じ A, B, A, B, A, C, A, B（初めての切り替えは B と C の2回、戻りは5回）
- 世界：T0〜T3 で判定、T4 とタイミングのずれ（T0 に p=0.1）は記録だけ。σ の下限は0.02 のまま（Physio2 の既定）
"""
SEALED_SEEDS = (9301, 9302, 9303, 9304)
SEALED_GROUPS = ((13.0, 48.0, 7.0), (13.0, 30.0, 7.0))
SEQ = ("A", "B", "A", "B", "A", "C", "A", "B")
SEALED_WORLDS = {
    "T0": dict(size=32, zone=24, v=0.25, noise=0.02, dwell=600),      # 標準（見ていないのは周期と乱数だけ）
    "T1": dict(size=32, zone=24, v=0.25, noise=0.02, dwell=1800),     # 滞在が長い（そうの持ち越し）
    "T2": dict(size=40, zone=32, v=0.25, noise=0.02, dwell=600),      # 格子40×40（ノイズの区域は右下8×8）
    "T3": dict(size=32, zone=24, v=0.25, noise=0.025, dwell=600),     # ノイズ0.025
    "T4": dict(size=40, zone=32, v=0.25, noise=0.025, dwell=1800),    # 全部まとめて（記録だけ）
}
JUDGED = ("T0", "T1", "T2", "T3")
JITTER = dict(world="T0", p=0.1)                                       # タイミングのずれ（原則2。記録だけ）

# 試運転（封印の材料に触れない。DESIGN.md 10.3 の足し2）：開発の周期と開発の乱数、封印でない世界の数
DRY_SEEDS = (300, 301, 302, 303)
DRY_GROUPS = ((12.0, 24.0, 8.0), (12.0, 20.0, 8.0))
DRY_WORLDS = {
    "D0": dict(size=32, zone=24, v=0.25, noise=0.02, dwell=600),
    "D1": dict(size=32, zone=24, v=0.25, noise=0.02, dwell=2000),     # 滞在の長い世界は、封印の1800 より少し長くして、書き込みの数を控えめ側に見る
    "D2": dict(size=36, zone=28, v=0.25, noise=0.02, dwell=600),
    "D3": dict(size=32, zone=24, v=0.25, noise=0.022, dwell=600),
    "D4": dict(size=36, zone=28, v=0.25, noise=0.022, dwell=2000),
}
DRY_JUDGED = ("D0", "D1", "D2", "D3")
DRY_JITTER = dict(world="D0", p=0.1)

SEALED_PERIODS = frozenset(p for g in SEALED_GROUPS for p in g)


def check_dev_groups(groups):
    """開発で封印の周期（13・48・7・30）を使おうとしたら止める（DESIGN.md 10.3 のコードの守り。evaluate.run が sealed=False のときに呼ぶ）"""
    bad = sorted({float(p) for g in groups for p in g} & SEALED_PERIODS)
    if bad:
        raise ValueError(f"周期 {bad} は Step 3 の封印テスト用なので、開発では使えない（DESIGN.md 10.3）")
