"""
描画：見た目と発火を並べたGIF、コマを抜き出したコンタクトシート、ロスと発火の推移のグラフ。
（チャットに貼るとGIFは最初のコマしか見えないので、確認はコンタクトシートで行う）
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 色（dataviz の参照パレット）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FIRE_RGB = np.array([0xEB, 0x68, 0x34], np.float32) / 255     # 発火はオレンジで描く

FONT_PATHS = ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc"]
PHASE_JA = {"grow": "成長", "maintain": "維持", "recover": "回復"}


def _font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rgba_to_rgb(rgba):
    """(4,H,W) の乗算済みアルファを白背景に合成して (H,W,3) の 0〜1 にする（nca.py の to_rgb と同じ式）"""
    rgba = np.clip(rgba, 0, 1)
    rgb = 1.0 - rgba[3:4] + rgba[:3]
    return np.clip(rgb, 0, 1).transpose(1, 2, 0)


def fire_panel(rgba, fire_rate):
    """花をうすく描いた上に、発火の割合（0〜1）をオレンジで重ねる"""
    rgb = rgba_to_rgb(rgba)
    faint = 1.0 - 0.25 * (1.0 - rgb.mean(axis=2, keepdims=True))
    faint = np.repeat(faint, 3, axis=2)
    a = fire_rate[..., None]
    return faint * (1 - a) + FIRE_RGB * a


def _to_img(arr01, scale):
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def _frame_image(fr, scale, font, title=""):
    left = _to_img(rgba_to_rgb(fr["rgba"]), scale)
    right = _to_img(fire_panel(fr["rgba"], fr["fire_rate"]), scale)
    pad, cap = 6, 26
    w = left.width * 2 + pad * 3
    h = left.height + pad * 2 + cap
    img = Image.new("RGB", (w, h), SURFACE)
    img.paste(left, (pad, pad + cap))
    img.paste(right, (pad * 2 + left.width, pad + cap))
    d = ImageDraw.Draw(img)
    d.text((pad, 4), f"{title}  t={fr['t'] + 1}  {PHASE_JA[fr['phase']]}", fill=INK, font=font)
    d.text((pad * 2 + left.width, 4), "発火", fill=INK_2, font=font)
    return img


def save_gif(frames, path, scale=2, title="", duration=60):
    font = _font(14)
    imgs = [_frame_image(fr, scale, font, title) for fr in frames if fr["gif"]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0, optimize=True)


def save_sheet(frames, steps, path, scale=2, title=""):
    """指定したステップのコマを、上段に見た目、下段に発火で並べる"""
    by_t = {fr["t"]: fr for fr in frames}
    cols = [by_t[t] for t in steps if t in by_t]
    font, font_title = _font(13), _font(16)
    cell = 72 * scale
    pad, head, lab = 6, 28, 20
    w = pad + len(cols) * (cell + pad)
    h = head + 2 * (lab + cell + pad)
    img = Image.new("RGB", (w, h), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((pad, 4), title, fill=INK, font=font_title)
    for i, fr in enumerate(cols):
        x = pad + i * (cell + pad)
        for row, (arr, name) in enumerate([(rgba_to_rgb(fr["rgba"]), "見た目"),
                                           (fire_panel(fr["rgba"], fr["fire_rate"]), "発火")]):
            y = head + row * (lab + cell + pad)
            d.text((x, y), f"t={fr['t'] + 1} {PHASE_JA[fr['phase']]}" if row == 0 else name,
                   fill=INK_2, font=font)
            img.paste(_to_img(arr, scale), (x, y + lab))
    img.save(path)


def save_curves(rows, proto, path, title=""):
    """ロス（対数）と、ステップごとの発火数を上下2枚に分けて描く（軸は1枚に1つ）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    t = np.array([r["t"] + 1 for r in rows])
    loss = np.array([r["loss"] for r in rows])
    fires = np.array([r["fires"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, facecolor=SURFACE)
    for ax, y, name in [(axes[0], loss, "ロス（RGBAの二乗誤差、対数）"),
                        (axes[1], fires, "発火数／ステップ（生きているセルの中）")]:
        ax.set_facecolor(SURFACE)
        ax.plot(t, y, color=SERIES[0], linewidth=1.5)
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(AXIS)
        for x0, lab in [(proto.grow_end, "成長→維持"), (proto.damage_at, "傷")]:
            ax.axvline(x0, color=MUTED, linewidth=0.8, linestyle="--")
        if ax is axes[0]:
            ax.set_yscale("log")
            ymax = ax.get_ylim()[1]
            for x0, lab in [(proto.grow_end, "成長→維持"), (proto.damage_at, "傷")]:
                ax.text(x0 + 4, ymax, lab, va="top", fontsize=8, color=INK_2)
    axes[1].set_xlabel("ステップ", color=INK_2, fontsize=9)
    axes[1].set_ylim(bottom=0)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
