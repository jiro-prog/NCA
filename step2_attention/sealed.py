"""
封印テストの定義（DESIGN.md 7.4。2026-09-29、そうの判断。走らせる前に決めた）。

卒業のときに1回だけ回す。開発のコードは check_dev_seed で、封印のシード（9000以上）を拒む。
"""
SEALED_TRAIN_SEEDS = (9001, 9002, 9003, 9004)     # 卒業判定の学習（レシピはそのまま、シードだけ封印のもの）
SEALED_EVAL_SEEDS = (9101, 9102, 9103, 9104)      # 評価の世界の乱数
SEALED_CHANGES = ((11.0, 19.0), (26.0, 8.0), (14.0, 40.0))   # 見せていない周期への、見える変化
SEALED_WORLDS = {
    "S0": dict(),                                  # 標準の世界（32×32、v=0.25、ノイズ0.005）
    "S1": dict(size=40, zone=32),                  # 格子の大きさ（ノイズの区域は右下8×8のまま）
    "S2": dict(v=0.2),                             # 波の速さ（5ステップに1列）
    "S3": dict(noise=0.006),                       # ノイズ（σ の下限は0.005のまま）
    "S4": dict(size=40, zone=32, v=0.2, noise=0.006),   # 全部まとめて（記録するが判定には使わない）
}
SEALED_MIN = 9000


def check_dev_seed(seed, what="シード"):
    """開発で封印のシードを使おうとしたら止める"""
    if seed is not None and int(seed) >= SEALED_MIN:
        raise ValueError(f"{what} {seed} は封印テスト用（{SEALED_MIN}以上）なので、開発では使えない（DESIGN.md 7.4）")
