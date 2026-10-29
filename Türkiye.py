import math
import os
import argparse
import time

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

IMAGE_PATH = "Türkiye.png"
OUT_PATH = ""

BG = (0, 0, 0)
NEON_GLOW = 0.0
NEON_RADIUS = 0.0025

TEXT_PHRASE = "TÜRKİYE  "
TEXT_FONT = ""
FONT_CANDIDATES = ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", "georgiab.ttf", "Arial Bold.ttf",
                   "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Poppins-Bold.ttf"]
TRACK = 0.12
LETTER_HEIGHT = 0.64

PITCH_DIV = 88
START_TURNS = 0.0
CENTER = (0.50, 0.50)
SS = 3
BASE_FONT = 160

DURATION = 14.0
SPEED_RATIO = 3.0

WIN_H_FRAC = 0.92
SIDE_MARGIN = 0.05

WINDOW_TITLE = "Mustafa Kemal ATATÜRK"


def load_source(path):
    with Image.open(path) as im:
        rgba = ImageOps.exif_transpose(im).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*BG, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def image_aspect(path):
    im = load_source(path)
    return im.width / im.height


_FONT = []


def load_font(size):
    if not _FONT:
        here = os.path.dirname(os.path.abspath(__file__))
        found = None
        names = ([TEXT_FONT] if TEXT_FONT else []) + FONT_CANDIDATES
        for name in names:
            for p in (name, os.path.join(here, name)):
                try:
                    ImageFont.truetype(p, 20)
                    found = p
                    break
                except Exception:
                    pass
            if found:
                break
        _FONT.append(found)
    p = _FONT[0]
    if p:
        return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def build_glyphs(phrase):
    f = load_font(BASE_FONT)
    capb = float(-f.getbbox("I", anchor="ls")[1])
    asc, desc = f.getmetrics()
    pad = int(BASE_FONT * 0.08)
    hc = int(2 * max(asc - capb / 2, desc + capb / 2)) + 2 * pad
    base, adv = {}, {}
    for ch in set(phrase):
        aw = float(f.getlength(ch))
        adv[ch] = aw
        if ch.strip() == "":
            continue
        img = Image.new("L", (int(math.ceil(aw)) + 2 * pad, hc), 0)
        ImageDraw.Draw(img).text((pad, hc / 2 + capb / 2), ch, font=f, fill=255, anchor="ls")
        base[ch] = img
    return base, adv, capb


def prepare(W, H, pitch, path):
    return load_source(path).resize((W, H), Image.Resampling.LANCZOS)


class Spiral:

    def __init__(self, W, H, path=IMAGE_PATH):
        self.W, self.H = W, H
        pitch = min(W, H) / PITCH_DIV
        color_map = prepare(W, H, pitch, path).resize(
            (W * SS, H * SS), Image.Resampling.BICUBIC)
        cx, cy = CENTER[0] * W, CENTER[1] * H
        self.pitch, self.cx, self.cy = pitch, cx, cy

        base, adv_base, capb = build_glyphs(TEXT_PHRASE)
        phrase = TEXT_PHRASE
        two_pi = 2 * math.pi
        k_arc = pitch / (4 * math.pi)
        th0 = two_pi * START_TURNS
        rmax = max(math.hypot(cx - px, cy - py) for px in (0, W) for py in (0, H)) + pitch
        th_max = two_pi * rmax / pitch

        def pos(s):
            th = math.sqrt(th0 * th0 + s / k_arc)
            r = pitch * th / two_pi
            return th, r, cx + r * math.cos(th), cy + r * math.sin(th)

        self.glyphs = []
        self.pen_xy = []
        s, ci, adv_guess = 0.0, 0, pitch * 0.3
        while True:
            ch = phrase[ci % len(phrase)]
            th, r, x, y = pos(s + adv_guess / 2)
            if th > th_max:
                break
            ix, iy = int(x), int(y)
            if not (0 <= ix < W and 0 <= iy < H):
                s += pitch * 0.35
                ci += 1
                continue

            hcap = pitch * LETTER_HEIGHT
            k = hcap / capb
            adv = adv_base[ch] * k + TRACK * hcap
            adv_guess = adv

            if ch in base:
                th, r, x, y = pos(s + adv / 2)
                rp = pitch / two_pi
                dx = rp * math.cos(th) - r * math.sin(th)
                dy = rp * math.sin(th) + r * math.cos(th)
                rot = -math.degrees(math.atan2(dy, dx))

                kk = k * SS
                src = base[ch]
                nw, nh = max(1, round(src.width * kk)), max(1, round(src.height * kk))
                m = src.resize((nw, nh), Image.LANCZOS).rotate(rot, resample=Image.BICUBIC, expand=True)

                left = int(round(x * SS - m.width / 2))
                top = int(round(y * SS - m.height / 2))
                tile = color_map.crop((left, top, left + m.width, top + m.height))
                self.glyphs.append((left, top, m, tile))
                self.pen_xy.append((x, y))
            s += adv
            ci += 1

        self.n = len(self.glyphs)
        self.T = DURATION

    def index_at(self, t):
        t = min(max(t, 0.0), self.T)
        frac = (t + 0.5 * (SPEED_RATIO - 1) * t * t / self.T) / (self.T * (1 + SPEED_RATIO) / 2)
        return min(int(self.n * frac), self.n)

    def draw(self, canvas, i0, i1):
        if i1 <= i0:
            return
        for left, top, m, col in self.glyphs[i0:i1]:
            canvas.paste(col, (left, top), m)

    def pen(self, i):
        if self.n == 0:
            return None
        return self.pen_xy[min(max(i - 1, 0), self.n - 1)]


def neon(img):
    if NEON_GLOW <= 0:
        return img
    r = max(1.0, img.height * NEON_RADIUS)
    bloom = img.reduce(3).filter(ImageFilter.GaussianBlur(r)).resize(img.size, Image.BILINEAR)
    bloom = bloom.point(lambda v: int(v * NEON_GLOW))
    return ImageChops.add(img, bloom)


def output_path(path):
    return OUT_PATH or os.path.splitext(os.path.abspath(path))[0] + "_spiral.png"


def save_png(path_in, path_out=None, width=2600):
    path_out = path_out or output_path(path_in)
    W = width
    H = round(W / image_aspect(path_in))
    sp = Spiral(W, H, path_in)
    canvas = Image.new("RGB", (W * SS, H * SS), BG)
    sp.draw(canvas, 0, sp.n)
    neon(canvas.resize((W, H), Image.LANCZOS)).save(path_out)
    print("Saved:", path_out)


class App:
    def __init__(self, path):
        import tkinter as tk
        from PIL import ImageTk
        self.tk, self.ImageTk = tk, ImageTk
        self.path = path
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="black")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        aspect = image_aspect(path)
        self.H = int(min(sh * WIN_H_FRAC, sw * 0.96 / (aspect * (1 + 2 * SIDE_MARGIN))))
        self.W = int(self.H * aspect)
        self.WW = min(int(self.W * (1 + 2 * SIDE_MARGIN)), int(sw * 0.96))
        x, y = (sw - self.WW) // 2, max(0, (sh - self.H) // 2 - 24)
        self.root.geometry(f"{self.WW}x{self.H}+{x}+{y}")

        self.cv = tk.Canvas(self.root, width=self.WW, height=self.H, bg="black",
                            highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.msg = self.cv.create_text(self.WW // 2, self.H // 2, text="preparing...",
                                       fill="#555555", font=("Helvetica", 14))
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<s>", lambda e: save_png(self.path))
        self.root.bind("<r>", lambda e: self.start())
        self.root.update()

        self.sp = Spiral(self.W, self.H, path)
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (self.W, self.H), BG))
        self.cv.delete(self.msg)
        self.item = self.cv.create_image(self.WW // 2, self.H // 2, image=self.photo)
        self.cv.bind("<Configure>", self.recenter)
        self.running = False
        self.start()

    def recenter(self, e):
        self.cv.coords(self.item, e.width // 2, e.height // 2)

    def start(self):
        self.big = Image.new("RGB", (self.W * SS, self.H * SS), BG)
        self.drawn = 0
        self.t0 = time.perf_counter()
        if not self.running:
            self.running = True
            self.tick()

    def tick(self):
        sp = self.sp
        t = time.perf_counter() - self.t0
        done = t >= sp.T
        i1 = sp.n if done else sp.index_at(t)
        sp.draw(self.big, self.drawn, i1)
        self.drawn = i1

        frame = neon(self.big.reduce(SS))
        if not done:
            p = sp.pen(i1)
            if p:
                rad = 3.0
                ImageDraw.Draw(frame).ellipse((p[0] - rad, p[1] - rad, p[0] + rad, p[1] + rad),
                                              fill=(255, 255, 255))
        self.photo.paste(frame)

        if done:
            self.running = False
        else:
            self.root.after(16, self.tick)

    def run(self):
        self.root.mainloop()


def choose_image():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
    finally:
        root.destroy()


def main():
    global TEXT_PHRASE, OUT_PATH
    parser = argparse.ArgumentParser(description="Turn a photo into spiral text art.")
    parser.add_argument("image", nargs="?", help="Path to the photo")
    parser.add_argument("--png", action="store_true", help="Save a PNG instead of playing the animation")
    parser.add_argument("--text", default=TEXT_PHRASE, help="Text to repeat")
    parser.add_argument("--output", default=OUT_PATH, help="Output PNG file")
    parser.add_argument("--width", type=int, default=2600, help="PNG width in pixels")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Text cannot be empty.")
    if args.width < 100:
        parser.error("Width must be at least 100 pixels.")
    TEXT_PHRASE = args.text
    OUT_PATH = args.output
    path = args.image or IMAGE_PATH
    if path and not os.path.isabs(path) and not os.path.isfile(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not path:
        path = choose_image()
    if not path:
        return
    if not os.path.isfile(path):
        parser.error("Photo not found: " + path)
    if os.path.realpath(output_path(path)) == os.path.realpath(path):
        parser.error("The output path cannot be the same as the source photo.")
    try:
        if args.png:
            save_png(path, width=args.width)
        else:
            App(path).run()
    except (OSError, ValueError) as exc:
        parser.exit(1, "Could not process the photo: " + str(exc) + "\n")


if __name__ == "__main__":
    main()