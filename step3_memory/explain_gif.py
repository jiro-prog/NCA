"""
人に説明するための GIF（2026-10-03、そうの頼み）。開発の世界で、床・眠る段・記憶の段を同じ世界に並べて回し、
予測の外れと書き込み（驚いて書き換え／眠って見直し／思い出し）を見せる。判定には使わない。
封印の世界・乱数・周期には触れない（開発の守り check_dev_seed・check_dev_groups を通す）。CPU で回す。

    .venv/Scripts/python.exe step3_memory/explain_gif.py

世界：標準の世界（32×32、波の速さ0.25、ノイズ0.02、右下8×8 はでたらめに光る区域）、評価の乱数300、組 12/24/8、並び A, B, A, B, A, C, A, B、滞在600、育ちから。
世界の実現は1つ（3つの段に同じセンサーの値を入れる）。
出力：step3_memory/runs/explain/（step3_explain.gif、sheet.png）。GIF は git に入れない
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step2_attention"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cell as C  # noqa: E402
import world as W  # noqa: E402
from common.viz import AXIS, GRID, INK, INK_2, MUTED, SURFACE, _font  # noqa: E402
from sealed import check_dev_seed  # noqa: E402
from sealed3 import check_dev_groups  # noqa: E402

SEED = 300
GROUP = (12.0, 24.0, 8.0)
SEQ = ("A", "B", "A", "B", "A", "C", "A", "B")
DWELL = 600
# 段：名前、副題、眠るか、記憶を持つか、線の色
RUNGS = (("床", "Step 2 のまま", False, False, "#898781"),
         ("眠る段", "＋睡眠", True, False, "#4a3aa7"),
         ("記憶の段", "＋睡眠＋記憶", True, True, "#e87ba4"))
KINDS = ("surprise", "sleep", "recall")                       # 描く順（思い出しは驚きの発火の中で起きるので、いちばん上に描く）
EV_RGB = {"surprise": (0xEB, 0x68, 0x34), "sleep": (0x2A, 0x78, 0xD6), "recall": (0x1B, 0xAF, 0x7A)}
EV_JA = {"surprise": "驚いて書き換え", "sleep": "眠って見直し", "recall": "思い出し"}
ERR_RGB = (0xE3, 0x49, 0x48)
FADE = 40              # 書き込みの残光（ステップ）。コマを間引いても見えるように
EMA = 16               # 外れのパネルは、セルごとに約16ステップでなめらかにした |外れ| を描く（ノイズの粒を消す）
ERR_LO, ERR_HI = 0.045, 0.10   # この範囲を白→赤にする。当たっている予測器の外れはノイズ並みで約0.032〜0.038（ノイズ0.02）なので白に近い
SCALE = 4
SLOW_LEN, SLOW_STEP, FAST_STEP = 170, 8, 100  # 切り替えの直後170ステップ（波が右端に届くまで124）はゆっくり、あとは早送り


def sample_times():
    ts = []
    for j in range(len(SEQ)):
        t0 = j * DWELL
        ts += [(t, False) for t in range(t0, t0 + SLOW_LEN, SLOW_STEP)]
        ts += [(t, True) for t in range(t0 + SLOW_LEN, t0 + DWELL, FAST_STEP)]
    ts.append((len(SEQ) * DWELL - 1, True))
    return ts


def simulate():
    check_dev_seed(SEED, "評価の乱数")
    check_dev_groups((GROUP,))
    wcfg = W.WorldConfig()
    S = wcfg.size
    per = dict(zip("ABC", GROUP))
    g = torch.Generator().manual_seed(SEED)
    ws = W.reset(wcfg, torch.tensor([per[SEQ[0]]], dtype=torch.float64), n=1, generator=g)
    cells = [C.Cell3(sleep=s, memory=m) for _, _, s, m, _ in RUNGS]
    sts = [C.init_state(1, S, S) for _ in RUNGS]
    wave = (~W.zone_mask(wcfg, "cpu")).numpy()
    T = len(SEQ) * DWELL
    last = np.full((len(RUNGS), len(KINDS), S, S), -10 ** 9)
    ema = np.zeros((len(RUNGS), S, S))
    err_series = np.zeros((len(RUNGS), T))
    writes = np.zeros((len(RUNGS), T))
    samples = dict(sample_times())
    frames = []
    for t in range(T):
        j = t // DWELL
        if t % DWELL == DWELL - 1 and j + 1 < len(SEQ):
            ws = W.set_period(ws, torch.tensor([per[SEQ[j + 1]]], dtype=torch.float64))
        sense = W.sense(ws, wcfg, g)
        errs = []
        for i, cell in enumerate(cells):
            sts[i], rec = cell.step(sts[i], sense)
            e = rec["e"][0, 0].abs().numpy()
            ema[i] += (e - ema[i]) / EMA
            errs.append(ema[i])
            err_series[i, t] = e[wave].mean()
            for k, kind in enumerate(KINDS):
                m = rec[kind][0, 0].numpy()
                last[i, k][m] = t
            writes[i, t] = float(rec["surprise"].float().mean() + rec["sleep"].float().mean())
        if t in samples:
            glow = np.clip(1 - (t - last) / FADE, 0, 1)
            frames.append(dict(t=t, fast=samples[t], world=sense[0, 0].numpy().copy(), err=[x.copy() for x in errs], glow=glow.copy()))
        ws = W.advance(ws)
    return dict(frames=frames, err=err_series, writes=np.cumsum(writes, 1), wave=wave, S=S, T=T, wcfg=wcfg)


def panel_world(x):
    v = np.clip(x, 0, 1)
    return np.stack([v, v, v], -1)


def panel_err(e):
    a = np.clip((e - ERR_LO) / (ERR_HI - ERR_LO), 0, 1)[..., None]
    white = np.array([0xFC, 0xFC, 0xFB]) / 255
    return white * (1 - a) + np.array(ERR_RGB) / 255 * a


def panel_events(world, glow):
    img = np.full(world.shape + (3,), 0.93)
    for k, kind in enumerate(KINDS):
        a = glow[k][..., None]
        img = img * (1 - a) + np.array(EV_RGB[kind]) / 255 * a
    return img


def to_img(arr):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def caption(t):
    """2行の説明（この世界の実現で確かめた動きだけを書く：explain_gif の確かめの数と、切り替えごとの反応・思い出しの数）"""
    j = t // DWELL
    p = SEQ[j]
    per = dict(zip("ABC", GROUP))
    if t - j * DWELL >= SLOW_LEN:
        return ("静かな時間（早送り）",
                "眠る段と記憶の段は、静かなときにときどき見直す（青）。床は何もしない")
    if j == 0:
        return (f"はじめは何も知らない（予測器0）。模様 {p}（周期{per[p]:g}）を見て学ぶ",
                "床は、育ったあとの係数が少し不正確なまま。眠る段と記憶の段は、見直して直す")
    if j == 1:
        return (f"初めての模様 {p}（周期{per[p]:g}）に変わる。波が左から届き、驚いて書き換える（オレンジ）",
                "床が驚いたのは約7割のセル。記憶の段は、最初の驚きで前の模様を思い出しかけ（緑）、すぐ書き換える")
    if p not in SEQ[:j]:
        return (f"初めての模様 {p}（周期{per[p]:g}）に変わる。波が左から届き、驚いて書き換える（オレンジ）",
                "床はほとんど驚かない（外れたまま）。眠る段と記憶の段は全部のセルが驚く（記憶の段は前の模様を思い出しかけることも）")
    return (f"前に見た模様 {p}（周期{per[p]:g}）に戻る",
            "床は慣れてしまい、ほとんど驚かない（外れたまま）。記憶の段は、しまっておいた係数を思い出す（緑）")


def draw_chart(d, box, series, t, T, title, ymax, fonts, fmt):
    x0, y0, x1, y1 = box
    f_s = fonts["s"]
    d.text((x0, y0 - 16), title, fill=INK_2, font=f_s)
    d.rectangle(box, outline=AXIS, fill=SURFACE)
    for j in range(1, len(SEQ)):
        xx = x0 + (x1 - x0) * j * DWELL / T
        d.line([(xx, y0), (xx, y1)], fill=GRID)
    for j, p in enumerate(SEQ):
        xx = x0 + (x1 - x0) * (j + 0.5) * DWELL / T
        d.text((xx - 4, y0 + 1), p, fill=MUTED, font=f_s)
    n = int((x1 - x0) * (t + 1) / T)
    for i, (_, _, _, _, col) in enumerate(RUNGS):
        pts = []
        for px in range(max(n, 1)):
            tt = min(int(px * T / (x1 - x0)), t)
            v = series[i][tt]
            pts.append((x0 + px, y1 - (y1 - y0) * min(v / ymax, 1.0)))
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
        d.text((x1 + 6, y0 + 2 + i * 15), f"{RUNGS[i][0]} {fmt(series[i][t])}", fill=col, font=f_s)
    xx = x0 + (x1 - x0) * (t + 1) / T
    d.line([(xx, y0), (xx, y1)], fill=INK_2)


def render(sim):
    S, T = sim["S"], sim["T"]
    P = S * SCALE
    fonts = dict(t=_font(17), m=_font(14), s=_font(12))
    M, G = 12, 16
    col_x = [M + c * (P + G) for c in range(4)]
    W_img = col_x[-1] + P + M + 110
    y_head = 8
    y_tl = 40
    y_title = 74
    y_p1 = y_title + 56
    y_p2 = y_p1 + P + 24
    y_c1 = y_p2 + P + 36
    y_c2 = y_c1 + 86
    y_cap = y_c2 + 70
    H_img = y_cap + 64
    # 誤差の推移は10ステップの移動平均で描く
    k = 10
    err_s = np.stack([np.convolve(e, np.ones(k) / k, mode="full")[:T] for e in sim["err"]])
    imgs, durs = [], []
    zone = sim["wcfg"].zone
    for fr in sim["frames"]:
        t = fr["t"]
        j = t // DWELL
        im = Image.new("RGB", (W_img, H_img), SURFACE)
        d = ImageDraw.Draw(im)
        d.text((M, y_head), "驚いたときだけ書き換えるセル　Step 3：眠りと記憶で、学び続けられるか", fill=INK, font=fonts["t"])
        # 並びのタイムライン
        x0, x1 = M, W_img - M - 110
        per = dict(zip("ABC", GROUP))
        for jj, p in enumerate(SEQ):
            a, b = x0 + (x1 - x0) * jj / len(SEQ), x0 + (x1 - x0) * (jj + 1) / len(SEQ)
            on = jj == j
            d.rectangle([a + 1, y_tl, b - 1, y_tl + 22], fill=("#0b0b0b" if on else GRID))
            d.text((a + 6, y_tl + 3), f"{p}（周期{per[p]:g}）", fill=(SURFACE if on else INK_2), font=fonts["s"])
        xx = x0 + (x1 - x0) * (t + 1) / T
        d.line([(xx, y_tl - 3), (xx, y_tl + 25)], fill="#e34948", width=2)
        d.text((x1 + 8, y_tl + 3), f"t = {t + 1:>4}", fill=INK, font=fonts["m"])
        # 列の見出し
        d.text((col_x[0], y_title), "世界（センサー）", fill=INK, font=fonts["m"])
        d.text((col_x[0], y_title + 18), "光の波が左から右へ", fill=INK_2, font=fonts["s"])
        for c, (name, sub, _, _, col) in enumerate(RUNGS):
            d.text((col_x[c + 1], y_title), name, fill=col, font=fonts["m"])
            d.text((col_x[c + 1], y_title + 18), sub, fill=INK_2, font=fonts["s"])
        # 世界
        wimg = to_img(panel_world(fr["world"]))
        im.paste(wimg, (col_x[0], y_p1))
        zx, zy = col_x[0] + zone * SCALE, y_p1 + zone * SCALE
        d.rectangle([zx, zy, col_x[0] + P - 1, y_p1 + P - 1], outline="#eda100", width=2)
        # 凡例（世界の下）
        ly = y_p2
        d.text((col_x[0], ly - 18), "見かた", fill=INK, font=fonts["m"])
        d.text((col_x[0], ly + 2), "予測の外れ：", fill=INK_2, font=fonts["s"])
        d.text((col_x[0], ly + 18), "白＝当たる 赤＝外れる", fill=INK_2, font=fonts["s"])
        for q, kind in enumerate(KINDS):
            yy = ly + 44 + q * 18
            d.rectangle([col_x[0], yy + 2, col_x[0] + 11, yy + 13], fill=EV_RGB[kind])
            d.text((col_x[0] + 16, yy), EV_JA[kind], fill=INK_2, font=fonts["s"])
        d.rectangle([col_x[0], ly + 102, col_x[0] + 11, ly + 113], outline="#eda100", width=2)
        d.text((col_x[0] + 16, ly + 100), "でたらめに光る区域", fill=INK_2, font=fonts["s"])
        d.text((col_x[0] + 16, ly + 116), "（外れても驚かない）", fill=INK_2, font=fonts["s"])
        # 段ごとの2つのパネル
        for c in range(len(RUNGS)):
            x = col_x[c + 1]
            d.text((x, y_p1 - 18), "予測の外れ", fill=INK_2, font=fonts["s"])
            im.paste(to_img(panel_err(fr["err"][c])), (x, y_p1))
            d.text((x, y_p2 - 18), "書き込み", fill=INK_2, font=fonts["s"])
            im.paste(to_img(panel_events(fr["world"], fr["glow"][c])), (x, y_p2))
        # 推移のグラフ
        cx0, cx1 = M, W_img - M - 110
        draw_chart(d, (cx0, y_c1, cx1, y_c1 + 56), err_s, t, T, "予測の外れ（波の区域の平均）", 0.10, fonts, lambda v: f"{v:.3f}")
        draw_chart(d, (cx0, y_c2, cx1, y_c2 + 56), sim["writes"], t, T, "書き込みの累計（1セルあたり。驚き＋睡眠）", max(float(sim["writes"].max()), 1.0), fonts, lambda v: f"{v:.1f}回")
        c1, c2 = caption(t)
        d.text((M, y_cap), c1, fill=INK, font=fonts["m"])
        d.text((M, y_cap + 20), c2, fill=INK_2, font=fonts["s"])
        d.text((M, y_cap + 42), "開発の世界の1例（標準の世界、評価の乱数300）。封印テストの結果ではない", fill=MUTED, font=fonts["s"])
        imgs.append(im)
        durs.append(80)
    durs[-1] = 4000
    return imgs, durs


def save_gif(imgs, durs, path):
    # 共通のパレット（コマごとに色がちらつかないように）
    pick = imgs[:: max(len(imgs) // 12, 1)]
    w, h = pick[0].size
    mont = Image.new("RGB", (w, h * len(pick)))
    for i, im in enumerate(pick):
        mont.paste(im, (0, h * i))
    pal = mont.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=False, disposal=1)


def save_sheet(imgs, frames, path, picks):
    ts = [fr["t"] for fr in frames]
    sel = [min(range(len(ts)), key=lambda i: abs(ts[i] - p)) for p in picks]
    w, h = imgs[0].size
    s = 0.5
    tw, th = int(w * s), int(h * s)
    cols = 4
    rows = (len(sel) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), SURFACE)
    for n, i in enumerate(sel):
        sheet.paste(imgs[i].resize((tw, th), Image.LANCZOS), ((n % cols) * tw, (n // cols) * th))
    sheet.save(path)


def main():
    out = HERE / "runs" / "explain"
    out.mkdir(parents=True, exist_ok=True)
    sim = simulate()
    imgs, durs = render(sim)
    save_gif(imgs, durs, out / "step3_explain.gif")
    picks = [120, 640, 760, 1240, 1840, 3040, 3640, 4799]
    save_sheet(imgs, sim["frames"], out / "sheet.png", picks)
    # 確かめ用の数（説明の文が実際の動きと合っているか）
    T = sim["T"]
    for i, (name, *_rest) in enumerate(RUNGS):
        e = sim["err"][i]
        late = [e[j * DWELL + 300: (j + 1) * DWELL].mean() for j in range(len(SEQ))]
        print(name, "各滞在の後半の外れ:", " ".join(f"{v:.3f}" for v in late), f"書き込みの累計 {sim['writes'][i][T - 1]:.1f}")
    print("コマ数", len(imgs), "大きさ", imgs[0].size, "GIF", round((out / "step3_explain.gif").stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
