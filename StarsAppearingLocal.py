#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate a local "_Stars Appearing_" sequence: sonification + animation </u>
#
# This builds the "_Stars Appearing_" piece from the "_Audible Universe_" planetarium
# show for **any site and any night**, and renders a matching animation.
#
# The sky is computed with `skyfield` from the _Hipparcos_ catalogue, sonified with
# `strauss`, and animated as an **equirectangular** (360 x 180 degree) panorama. A
# planetarium dome master is produced by letting `ffmpeg`'s `v360` filter reproject
# the panorama to fisheye - we never render fisheye ourselves.
#
# The single most important idea here is that the animation does **not** re-derive
# when each star appears. It reads the timings straight out of the rendered
# sonification via `strauss.get_table()`, so sound and picture cannot drift apart.
#
# The panorama the stars are drawn over is rendered here too, from NASA's all-sky
# star map, for the same site and instant - so there is no background image to
# supply, and no way for it to fall out of step with the sound.
#
# This module is deliberately *not* where the sonification happens. It supplies the
# things that are not `strauss` - the sky, from `skyfield`, and the animation and
# its backdrop, from `numpy` and `ffmpeg` - and the notebook beside it drives
# `strauss` itself, in the open, with the `stars_appearing` style carrying the
# recipe.
#
# Extra requirements beyond `strauss`:  `pip install skyfield`
# (and a working `ffmpeg` on your `PATH`).

import hashlib
import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tqdm



# Native size of each NASA star map, and so of a "full" render. Both are 2:1,
# the shape of a 360 x 180 degree panorama - a frame of any other shape holds
# the same sky stretched.
STARMAP_SIZES = {"4k": (4096, 2048), "8k": (8192, 4096)}

# Frame sizes small enough to render while you are still choosing a site and
# a night. 'full' is not here: it comes from the star map, above.
SIZES = {"fast_preview": (512, 256), "preview": (1024, 512)}


# <u> __Settings:__ </u>
#
# Everything you would want to change lives in one place. The defaults describe
# the Sherwood Observatory / Sherwood Planetarium site looking south.

@dataclass
class Config:
    """Every knob for one run of the sequence."""

    # -- where and when --------------------------------------------------
    # latitude is +ve north; longitude is +ve *east*, so 1.22 degrees west
    # of Greenwich is -1.22
    latitude: float = 53.1143737
    longitude: float = -1.2219389
    date_time: str = "2026-09-19 19:00:00"      # local wall-clock, YYYY-MM-DD HH:mm:ss
    time_zone: str = "Europe/London"            # TZ identifier, e.g. 'Europe/London'

    # which way is the listener/viewer facing? a cardinal point, and the
    # centre of both the panorama and the stereo/surround image
    facing: str = "S"

    # faintest star to include; larger numbers mean more, dimmer stars
    mag_limit: float = 5.0

    # -- the sound -------------------------------------------------------
    # passed to `strauss.sonify`, and used here to work out how many frames
    # the animation needs. The *sound itself* is chosen by the strauss style,
    # not from here.
    duration: float = 60.0                      # seconds
    system: str = "5.1"                         # 'mono', 'stereo', '5.1', 'ambix2', ...

    # -- the picture -----------------------------------------------------
    # frame size: either a name, or an explicit `(width, height)`.
    #   'full'         - the native size of the star map chosen below, so that
    #                    its pixels are used as they are rather than resampled
    #   'preview'      - 1024 x 512, for while you are still deciding
    #   'fast_preview' - 512 x 256, quicker still
    # The named sizes are all 2:1, matching a 360 x 180 degree panorama. Give
    # a pair of your own if you need some other shape; the sky will be
    # stretched to fill it, and the star pulses stretched to match.
    size: str | tuple = "full"
    fps: int = 30

    # filled in from `size` below
    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)

    # peak radius in pixels of the brightest and faintest star pulses
    max_star_px: float = 15.0
    min_star_px: float = 1.0

    # a star flashes rather than persists: it swells over tau_in and decays
    # over tau_out, both in seconds
    tau_in: float = 0.06
    tau_out: float = 0.6

    # opacity of a star at full size
    star_alpha: float = 0.6

    # an equirectangular panorama over-samples the sky near the poles, so
    # pulses are stretched in azimuth to stay round on the dome. Clamped,
    # since the stretch diverges at the zenith itself.
    max_stretch: float = 40.0

    # panorama to lay the stars over: a 360x180 degree image with `facing` at
    # its centre. One of
    #   "auto"  - render one to match every setting above, from NASA's all-sky
    #             star map, so that the pulses land on the stars drawn in it
    #   a path  - use that image as it is, e.g. `"my_sky.png"`
    #   None    - a plain black sky
    background: Path | str | None = "auto"

    # which NASA star map "auto" renders from. '4k' is a 36 MB download, '8k'
    # is 130 MB and only worth it past about 4096 pixels wide.
    starmap: str = "4k"

    # linear gain applied to that map before it is encoded for the screen.
    # Raise it to bring the Milky Way up, lower it to keep the sky dark and
    # let the star pulses carry the picture.
    sky_exposure: float = 0.75

    # leave everything below the horizon black, as the ground would. Nothing
    # is lost by it, since no star is sounded from down there - and it is
    # worth turning on for the dome master, where the sky beneath your feet
    # otherwise fills the corners and leaves no horizon to see.
    horizon: bool = False

    # -- outputs ---------------------------------------------------------
    outdir: Path = Path("stars_appearing_out")

    # which videos to write. One of
    #   'panorama' - the equirectangular 360 x 180 degree view
    #   'dome'     - the fisheye planetarium master
    #   'both'     - one of each, from a single pass of the frame generator
    output: str = "both"

    dome_pitch: float = 90.0                    # degrees to tilt the dome view up
    dome_size: int | None = None                # square edge; defaults to `height`

    # -- reproducibility -------------------------------------------------
    # magnitudes are jittered to break ties, and notes detuned very slightly
    seed: int = 0

    # where skyfield keeps its downloaded catalogue and ephemeris
    cache: Path = Path("~/.strauss_skyfield").expanduser()

    def __post_init__(self):
        if isinstance(self.size, str):
            sizes = {**SIZES, "full": STARMAP_SIZES[self.starmap]}
            if self.size not in sizes:
                raise ValueError(f"'{self.size}' is not a frame size. Choose "
                                 f"from {sorted(sizes)}, or give a "
                                 f"(width, height) pair.")
            self.width, self.height = sizes[self.size]
        else:
            self.width, self.height = self.size

        self.outdir = Path(self.outdir)
        if self.background not in (None, "auto"):
            self.background = Path(self.background)
        if self.dome_size is None:
            self.dome_size = self.height


# <u> __The sky:__ </u>
#
# `skyfield` gives us the altitude and azimuth of every catalogue star as seen
# from the chosen site at the chosen instant. We cut the catalogue down by
# magnitude *before* computing positions, which is the difference between
# transforming ~118,000 stars and ~1,500.

CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def unit_scale(values):
    """Scale values onto 0-1.

    A handful of stars can share a magnitude or a colour exactly - a
    single constellation, say - which would otherwise divide by a zero
    range. Those degenerate cases land in the middle of the scale.
    """
    values = np.asarray(values, dtype=float)
    low, high = values.min(), values.max()
    if not np.isfinite(high - low) or high == low:
        return np.full(values.shape, 0.5)

    return (values - low) / (high - low)


def facing_degrees(facing):
    """Bearing of a cardinal point in degrees clockwise from north."""
    if facing not in CARDINALS:
        raise ValueError(f"'{facing}' is not a cardinal point. "
                         f"Choose from: {CARDINALS}")
    return 360.0 * CARDINALS.index(facing) / len(CARDINALS)


def load_catalogue(cfg, loader):
    """Read the Hipparcos catalogue, keeping stars we could plausibly show."""
    from skyfield.data import hipparcos

    with loader.open(hipparcos.URL) as f:
        cat = pd.read_csv(
            f, sep="|", names=hipparcos._COLUMN_NAMES, compression=None,
            usecols=["HIP", "Vmag", "RAdeg", "DEdeg", "Plx", "pmRA", "pmDE", "B-V"],
            na_values=["     ", "       ", "        ", "            ", "      "],
        )

    cat.columns = ("hip", "magnitude", "ra_degrees", "dec_degrees",
                   "parallax_mas", "ra_mas_per_year", "dec_mas_per_year", "bv")
    cat = cat.assign(ra_hours=cat["ra_degrees"] / 15.0, epoch_year=1991.25)
    cat = cat.set_index("hip")

    # cut down before doing any astrometry - a star we will never draw or
    # sound is not worth transforming
    keep = (cat["magnitude"] < cfg.mag_limit) & np.isfinite(cat["bv"])

    return cat[keep]


def observed_sky(cfg):
    """Stars above the horizon at the configured place and time.

    Returns:
      sky (:obj:`pandas.DataFrame`): a row per visible star, indexed by
        Hipparcos number, with its altitude and azimuth in degrees,
        `V` magnitude, and `B-V` colour.
    """
    from skyfield.api import Loader, Star, wgs84

    loader = Loader(str(cfg.cache), verbose=True)
    cat = load_catalogue(cfg, loader)

    when = datetime.strptime(cfg.date_time, "%Y-%m-%d %H:%M:%S")
    when = when.replace(tzinfo=ZoneInfo(cfg.time_zone))
    t = loader.timescale().from_datetime(when)

    earth = loader("de421.bsp")["earth"]
    observer = (earth + wgs84.latlon(cfg.latitude, cfg.longitude)).at(t)

    alt, az, _ = observer.observe(Star.from_dataframe(cat)).apparent().altaz()

    sky = pd.DataFrame({"alt": alt.degrees,
                        "az": az.degrees,
                        "magnitude": cat["magnitude"].to_numpy(float),
                        "bv": cat["bv"].to_numpy(float)},
                       index=cat.index)

    sky = sky[np.isfinite(sky["alt"]) & (sky["alt"] > 0)].copy()

    # break ties so that no two stars land on exactly the same instant
    rng = np.random.default_rng(cfg.seed)
    sky["magnitude"] += 1e-2 * rng.random(len(sky))

    return sky.sort_values("magnitude")


def star_frame(sky, cfg):
    """The columns the `stars_appearing` style asks for, one row per star.

    The column names are the style's `input:` names - `sonify` matches a
    `DataFrame` to a style by name - and this is where the observer's own
    point of view is applied, which is the part no style file could do:

      - `azimuth` strauss measures **anticlockwise from straight ahead**,
        while astronomical azimuth runs **clockwise from north**, so it is
        the facing direction *minus* the star's azimuth.
      - `polar` is measured from the zenith down, not the horizon up.
      - `volume` quietens the dimmer stars. They are far more numerous, so
        without this the piece grows steadily louder as it goes.
      - `pitch_shift` detunes each note very slightly, so that the many
        stars sharing a note do not phase against one another.

    Args:
      sky (:obj:`pandas.DataFrame`): as `observed_sky` returns.

    Returns:
      frame (:obj:`pandas.DataFrame`): ready for `strauss.sonify`, indexed
        by `HIP<number>` so each note can be traced back to its star.
    """
    smag = unit_scale(sky["magnitude"].to_numpy(float))
    rng = np.random.default_rng(cfg.seed + 1)

    return pd.DataFrame({
        "magnitude": sky["magnitude"].to_numpy(float),
        "colour":    sky["bv"].to_numpy(float),
        "azimuth":   (facing_degrees(cfg.facing) - sky["az"].to_numpy(float)) % 360,
        "polar":     90.0 - sky["alt"].to_numpy(float),
        "volume":    (1 - smag) ** 0.5,
        "pitch_shift": 5e-3 * rng.random(len(sky)),
    }, index=[f"HIP{hip}" for hip in sky.index])


def sonified_events(frame):
    """What sounded and when, read back out of the rendered sonification.

    This is the join between sound and picture: the animation takes its
    timings from here rather than working them out again, so the two
    cannot drift apart.

    Args:
      frame (:obj:`pandas.DataFrame`): the frame that was sonified, for
        the magnitude and colour each pulse is drawn with.

    Returns:
      events (:obj:`pandas.DataFrame`): a row per note, in time order,
        with its time in seconds and its angles in degrees.
    """
    import strauss

    # the units sit in a second column level, which is for reading rather
    # than for arithmetic - drop to the plain names before touching the
    # numbers
    flat = strauss.get_table().copy()
    flat.columns = flat.columns.get_level_values(0)

    events = pd.DataFrame({
        "time":    flat["Time"].to_numpy(float),
        "azimuth": flat["Azimuthal Angle"].to_numpy(float),
        "polar":   flat["Polar Angle"].to_numpy(float),
    }, index=flat["Source"].to_numpy())

    return events.join(frame[["magnitude", "colour"]]).sort_values("time")


# <u> __The background sky:__ </u>
#
# The pulses read far better over a real sky than over black, and a panorama to
# lay them on can be rendered from the same site, instant and facing rather than
# supplied by hand. NASA's *Deep Star Maps* are all-sky equirectangular images in
# *celestial* coordinates, so turning one into the view from a given place at a
# given moment is a matter of asking, for every pixel of the output, which point
# of the sky it looks at - which is `skyfield` again, run backwards.
#
# The map is `OpenEXR`, decoded here by `ffmpeg`, which this example already
# requires. That keeps the extra dependencies at nil.

STARMAP_URL = ("https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/"
               "starmap_2020_{size}.exr")


def _download(url, path):
    """Fetch `url` to `path`, unless it is already there."""
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("content-length", 0))
        # write beside the target and rename, so an interrupted download does
        # not leave half a file behind to be trusted on the next run
        part = path.with_suffix(path.suffix + ".part")
        with open(part, "wb") as f, tqdm.tqdm(total=total, unit="B",
                                              unit_scale=True,
                                              desc=path.name) as bar:
            for chunk in iter(lambda: response.read(1 << 16), b""):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(path)

    return path


def _decode_exr(path):
    """Read an `OpenEXR` image into a float32 `(H, W, 3)` `RGB` array."""
    size = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
         str(path)], capture_output=True, text=True, check=True).stdout
    W, H = (int(n) for n in size.strip().split("x"))

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "gbrpf32le", "-"], capture_output=True, check=True).stdout

    # planar, and in ffmpeg's green-blue-red plane order
    planes = np.frombuffer(raw, dtype="<f4").reshape(3, H, W)

    return np.stack([planes[2], planes[0], planes[1]], axis=-1)


def _sample(image, x, y):
    """Bilinear sample of `image` at fractional `(x, y)`.

    Wraps around in `x`, since the map is a full turn of the sky and its
    left and right edges are the same meridian, and clamps in `y`, where
    the poles are the ends of the picture rather than a seam.
    """
    H, W = image.shape[:2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.clip(np.floor(y), 0, H - 2).astype(np.int32)
    tx = (x - x0).astype(np.float32)[..., None]
    ty = (y - y0).astype(np.float32)[..., None]
    x1, x0 = (x0 + 1) % W, x0 % W
    y1 = y0 + 1

    return (image[y0, x0] * ((1 - tx) * (1 - ty))
            + image[y0, x1] * (tx * (1 - ty))
            + image[y1, x0] * ((1 - tx) * ty)
            + image[y1, x1] * (tx * ty))


def sky_panorama(cfg, out_path=None):
    """Render the sky over the configured site as an equirectangular panorama.

    The result is laid out exactly as the animation is - `facing` down the
    middle, the zenith along the top edge and the nadir along the bottom -
    so that the star pulses land on the stars already drawn in it.

    Args:
      cfg (:obj:`Config`): the run this panorama is for. Its site, instant,
        facing, `width` and `height` all feed into the result.
      out_path (:obj:`pathlib.Path`, optional): where to write the `PNG`.
        Defaults to a name inside `cfg.outdir` carrying a digest of the
        settings that produced it, so that changing any of them gets you a
        new panorama rather than the last one.

    Returns:
      out_path (:obj:`pathlib.Path`): the panorama written, or the one
        already there.
    """
    from skyfield.api import Loader, wgs84

    settings = (cfg.latitude, cfg.longitude, cfg.date_time, cfg.time_zone,
                cfg.facing, cfg.width, cfg.height, cfg.starmap,
                cfg.sky_exposure, cfg.horizon)
    if out_path is None:
        digest = hashlib.sha1(repr(settings).encode()).hexdigest()[:8]
        out_path = cfg.outdir / f"sky_panorama_{digest}.png"
    out_path = Path(out_path)

    if out_path.exists():
        return out_path

    starmap = _download(STARMAP_URL.format(size=cfg.starmap),
                        cfg.cache / f"starmap_2020_{cfg.starmap}.exr")

    loader = Loader(str(cfg.cache), verbose=True)
    when = datetime.strptime(cfg.date_time, "%Y-%m-%d %H:%M:%S")
    when = when.replace(tzinfo=ZoneInfo(cfg.time_zone))
    t = loader.timescale().from_datetime(when)

    earth = loader("de421.bsp")["earth"]
    observer = (earth + wgs84.latlon(cfg.latitude, cfg.longitude)).at(t)

    # where each pixel of the panorama looks: `facing` at the centre column,
    # the zenith along the top row. Matching `render_frames`, which puts a
    # star of polar angle `p` at row `p * H / 180`, and of astronomical
    # azimuth `a` at column `(180 - facing + a) * W / 360`.
    azimuth = (facing_degrees(cfg.facing) - 180
               + np.linspace(0.0, 360.0, cfg.width)) % 360.0
    altitude = np.linspace(90.0, -90.0, cfg.height)
    az_grid, alt_grid = np.meshgrid(azimuth, altitude)

    ra, dec, _ = observer.from_altaz(alt_degrees=alt_grid.ravel(),
                                     az_degrees=az_grid.ravel()).radec()
    ra = ra._degrees.reshape(alt_grid.shape)
    dec = dec.degrees.reshape(alt_grid.shape)

    image = _decode_exr(starmap)
    src_h, src_w = image.shape[:2]

    # the map is the sky seen from outside, and we are underneath it, so
    # right ascension runs the other way
    sky = _sample(image,
                  (((-ra / 360.0) + 0.5) % 1.0) * (src_w - 1),
                  ((90.0 - dec) / 180.0) * (src_h - 1))

    # the map is in linear light; expose, then encode to something a screen
    # can show
    sky = np.clip(sky * cfg.sky_exposure, 0.0, 1.0) ** (1 / 2.2)
    sky = (sky * 255.0 + 0.5).astype(np.uint8)

    # no stars are sounded from below the horizon, so nothing is lost by
    # putting the ground there - and the dome master needs it, or the sky
    # beneath your feet fills its corners and swallows the horizon
    if cfg.horizon:
        sky[altitude < 0.0] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{cfg.width}x{cfg.height}", "-i", "pipe:0",
                    "-frames:v", "1", str(out_path)],
                   input=sky.tobytes(), check=True)

    return out_path


def resolve_background(cfg):
    """The panorama to lay the stars over, rendering one if asked to.

    Returns:
      path (:obj:`pathlib.Path` or :obj:`None`): the image to use, or
        `None` for a plain black sky.
    """
    if cfg.background is None:
        return None
    if cfg.background == "auto":
        return sky_panorama(cfg)

    return Path(cfg.background)


# <u> __The sound design:__ </u>
#
# The *Sonification Suite* keeps a sound design of its own for this piece, "Night
# Harp", and its instrument and chord are worth having here. Its style files are
# written in the Suite's own dialect rather than strauss's, so they cannot simply
# be loaded - but the two parts that make the sound, the samples and the notes,
# carry across as they are. The Suite stores each instrument as a directory of
# `.wav` files named by the note each one sounds, which is exactly what strauss's
# `Sampler` expects of a sample directory, so the folder can be handed straight to
# a style's `generator.sample`.

SUITE_REPO = "Audio-Universe/sonification-suite"
SUITE_SAMPLES = "src/backend/sound_assets/samples"

# the chord of the Suite's "Night Harp" style, for the harp to play
NIGHT_HARP_NOTES = ["F2", "C2", "F3", "A3", "E4", "G4"]


def suite_samples(name="Harp", cache=None, ref="main"):
    """Download one of the Suite's instruments, and return the folder.

    Fetched once and kept, so later runs cost nothing. The listing comes
    from GitHub's unauthenticated API, which is rate-limited per address -
    a concern only on the first run, since a folder already holding samples
    is used without asking.

    Args:
      name (`optional`, :obj:`str`): the instrument's folder in the Suite,
        e.g. `"Harp"`.
      cache (`optional`, :obj:`pathlib.Path`): where to keep it. Defaults
        to the same cache as the catalogue and the star map.

    Returns:
      folder (:obj:`pathlib.Path`): the directory of samples.
    """
    folder = Path(cache or Config.cache) / "suite_samples" / name

    if any(folder.glob("*.[wW][aA][vV]")):
        return folder

    listing = (f"https://api.github.com/repos/{SUITE_REPO}/contents/"
               f"{SUITE_SAMPLES}/{name}?ref={ref}")
    with urllib.request.urlopen(listing) as response:
        entries = json.load(response)

    wavs = [e for e in entries if e["name"].lower().endswith(".wav")]
    if not wavs:
        raise ValueError(f"the Suite has no samples for '{name}'.")

    folder.mkdir(parents=True, exist_ok=True)
    for entry in tqdm.tqdm(wavs, desc=f"{name} samples", unit="file"):
        target = folder / entry["name"]
        if target.exists():
            continue
        # write beside the target and rename, so an interrupted download
        # does not leave half a sample behind to be trusted on the next run
        part = target.with_name(target.name + ".part")
        with urllib.request.urlopen(entry["download_url"]) as response, \
                open(part, "wb") as f:
            shutil.copyfileobj(response, f)
        part.rename(target)

    return folder


def restyle(base="stars_appearing", sample=None, notes=None, name=None,
            description=None, out_path=None):
    """Write a copy of a strauss style, with a different instrument or chord.

    The recipe itself - what maps to what, and how the notes are shaped -
    is left alone, so the sonification still sounds one note per star and
    still carries the same data. Only the sound the notes are made of, and
    the chord they are drawn from, change.

    Args:
      base (`optional`, :obj:`str`): the style to start from, by name or
        path, as `strauss.sonify` takes it.
      sample (`optional`, :obj:`pathlib.Path` or :obj:`str`): instrument
        to play it on - a directory of samples, as `suite_samples`
        returns, a soundfont, or the name of a built-in.
      notes (`optional`, :obj:`list`): the chord to draw notes from.
      name (`optional`, :obj:`str`): what to call the result. Worth giving
        wherever the base style's name describes the sound it no longer
        has.
      description (`optional`, :obj:`str`): likewise for its description.
      out_path (`optional`, :obj:`pathlib.Path`): where to write the
        style. Defaults to `<base>_restyled.yml` in the working directory.

    Returns:
      out_path (:obj:`str`): the style written, ready to hand to
      `strauss.sonify` as its `style`.
    """
    import yaml
    from strauss.audio_figure import load_style

    style = load_style(base)

    if sample is not None:
        style.setdefault("generator", {})["sample"] = str(sample)
    if notes is not None:
        style["notes"] = list(notes)
    if name is not None:
        style["name"] = name
    if description is not None:
        style["description"] = description

    out_path = Path(out_path or f"{Path(base).stem}_restyled.yml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(style, sort_keys=False))

    return str(out_path)


# the sounds to choose between, as the notebooks offer them
SOUNDS = ["Night Harp", "Stars Appearing"]


def chosen_style(sound="Night Harp", cfg=None):
    """The style to sonify with, for one of the sounds in `SOUNDS`.

    `"Stars Appearing"` is the glockenspiel of the original planetarium
    piece, and is the `stars_appearing` style as it ships. `"Night Harp"`
    keeps that same recipe and swaps only the sound it is made of - the
    Suite's harp samples, fetched once, and the chord of its own "Night
    Harp" style - so every mapping is left as it is.

    Args:
      sound (`optional`, :obj:`str`): one of `SOUNDS`.
      cfg (`optional`, :obj:`Config`): for the sample cache, and for where
        a restyled style file is written.

    Returns:
      style (:obj:`str`): a style name or path, for `strauss.sonify`.
    """
    if sound not in SOUNDS:
        raise ValueError(f"'{sound}' is not a sound. Choose from {SOUNDS}.")

    if sound == "Stars Appearing":
        return "stars_appearing"

    cfg = cfg or Config()
    return restyle("stars_appearing",
                   sample=suite_samples("Harp", cfg.cache),
                   notes=NIGHT_HARP_NOTES,
                   name="Stars Appearing (Night Harp)",
                   description="Brightest stars appear first, pitch mapped to "
                               "colour. Harp and chord from the Sonification "
                               "Suite's 'Night Harp'.",
                   out_path=cfg.outdir / "stars_appearing_harp.yml")


# <u> __The animation:__ </u>
#
# Each star is a pulse that swells and fades as its note sounds. Frames are
# generated as raw `RGBA` and piped straight into `ffmpeg` - no intermediate
# `PNG`s, so nothing touches the disk between here and the finished video.
#
# Two things keep this quick. A star is only drawn while it is bigger than half
# a pixel, which for these envelopes is around a second out of the whole piece,
# so a binary search over the (sorted) event times finds the handful of stars
# actually alive in each frame. And only the rows those stars touch are cleared
# and converted, rather than the whole 4K canvas.

def star_rgb(bv01):
    """Colour of a star from its normalised `B-V`, 0 bluest to 1 reddest."""
    return 1 - 0.3 * (np.array([1.0, 0.5, 0.0]) - bv01) ** 2


def render_frames(events, cfg):
    """Yield one raw `RGBA` frame per frame of the animation.

    Args:
      events (:obj:`pandas.DataFrame`): a row per star, with the columns
        `time`, `azimuth` and `polar` taken from the sonification's own
        table - so that the picture cannot disagree with the sound - plus
        the `magnitude` and `colour` that were sonified, which set how big
        and what colour each pulse is.
    """
    W, H = cfg.width, cfg.height
    n_frames = int(round(cfg.duration * cfg.fps))

    t_star = events["time"].to_numpy(float)

    # horizontal pixel: strauss azimuth runs anticlockwise from the facing
    # direction, the image runs left to right, and `facing` sits at the centre
    cx = ((180.0 - events["azimuth"].to_numpy(float)) % 360.0) * (W / 360.0)
    cy = events["polar"].to_numpy(float) * (H / 180.0)

    # peak radius: bright stars are big, faint ones small
    brightness = 1 - unit_scale(events["magnitude"].to_numpy(float))
    amp = 1.2 * (cfg.max_star_px * brightness + cfg.min_star_px)

    # colour, clipped to the bulk of the B-V range so a few outliers do not
    # flatten everything else
    bv = events["colour"].to_numpy(float)
    bv01 = unit_scale(np.clip(bv, *np.percentile(bv, [1, 99])))
    rgb = star_rgb(bv01[:, None])

    # azimuthal stretch, so a pulse stays round on the sky - and so stays
    # round on the dome once reprojected. Two things stretch it. A pulse at
    # polar angle `t` spans 1/sin(t) times as much azimuth as it does
    # altitude, diverging at the poles. And the canvas carries 360 degrees
    # across but only 180 down, so unless it is 2:1 its pixels are not square
    # in angle, by a further factor of W/2H.
    aspect = W / (2.0 * H)
    stretch = np.clip(aspect / np.sin(np.pi * np.clip(cy, 0.5, H - 0.5) / H),
                      aspect, cfg.max_stretch)

    # how long before and after its note a star is worth drawing at all
    reach = np.log(max(2.0 * amp.max(), np.e))
    lead, trail = cfg.tau_in * reach, cfg.tau_out * reach

    # premultiplied colour and coverage, reused between frames
    colour = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    out = np.zeros((H, W, 4), dtype=np.uint8)

    dirty = (0, H)

    for frame in range(n_frames):
        now = frame / cfg.fps

        # forget the previous frame, but only where it drew something
        y0, y1 = dirty
        colour[y0:y1] = 0.0
        alpha[y0:y1] = 0.0

        lo = np.searchsorted(t_star, now - trail, side="left")
        hi = np.searchsorted(t_star, now + lead, side="right")

        touched_lo, touched_hi = H, 0

        for i in range(lo, hi):
            dt = now - t_star[i]
            env = np.exp(dt / cfg.tau_in) if dt < 0 else np.exp(-dt / cfg.tau_out)
            size = amp[i] * env
            if size < 0.5:
                continue

            sx = stretch[i]
            # a pulse near the zenith stretches a long way in azimuth. Keep it
            # under one full turn, so that no column appears twice once wrapped
            # - a repeat would silently drop one of its contributions.
            rx = min(size * sx, (W - 3) / 2)

            ys0 = max(int(np.floor(cy[i] - size)), 0)
            ys1 = min(int(np.ceil(cy[i] + size)) + 1, H)
            if ys1 <= ys0:
                continue

            xs = np.arange(int(np.floor(cx[i] - rx)),
                           int(np.ceil(cx[i] + rx)) + 1)[:W]
            ys = np.arange(ys0, ys1)

            # radius in units of the (unstretched) pulse
            dx = (xs - cx[i]) / sx
            dy = ys - cy[i]
            r = np.hypot(dx[None, :], dy[:, None])

            # a one-pixel soft edge, standing in for cairo's antialiasing
            cover = np.clip(size + 0.5 - r, 0.0, 1.0).astype(np.float32)
            if not cover.any():
                continue

            idx = np.ix_(ys, xs % W)          # wrap around the back of the sky

            # composite the pulse over whatever is already there
            have = alpha[idx]
            add = cover * cfg.star_alpha * (1.0 - have)
            colour[idx] += rgb[i] * add[..., None]
            alpha[idx] = have + add

            touched_lo, touched_hi = min(touched_lo, ys0), max(touched_hi, ys1)

        # everything the stars did not touch stays fully transparent, so the
        # background shows through untouched
        out[:] = 0

        if touched_hi > touched_lo:
            sl = slice(touched_lo, touched_hi)
            a = alpha[sl]
            # undo the premultiplication, since ffmpeg expects straight alpha
            np.multiply(colour[sl], 255.0 / np.maximum(a, 1e-6)[..., None],
                        out=colour[sl])
            out[sl, :, :3] = np.clip(colour[sl], 0, 255).astype(np.uint8)
            out[sl, :, 3] = (a * 255).astype(np.uint8)

        dirty = (touched_lo, touched_hi) if touched_hi > touched_lo else (0, 0)

        yield out.tobytes()


# <u> __Compositing:__ </u>
#
# One `ffmpeg` pass does the lot: lay the star frames over the background
# panorama, mux the rendered audio, and - for the dome master - reproject the
# equirectangular result to fisheye with the `v360` filter.

def encode(cfg, audio, out_path, sky=None, dome=False):
    """Build the ffmpeg command for one output, and return it.

    Args:
      sky (:obj:`pathlib.Path`, optional): the panorama to lay the stars
        over, as returned by `resolve_background`. `None` gives black.
    """
    if sky is not None:
        background = ["-loop", "1", "-framerate", str(cfg.fps), "-i", str(sky)]
    else:
        background = ["-f", "lavfi", "-i",
                      f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}"]

    chain = (f"[0:v]scale={cfg.width}:{cfg.height},setsar=1,format=rgba[bg];"
             "[bg][1:v]overlay=shortest=1[comp];[comp]")
    if dome:
        # equirectangular in, fisheye out, tilted up so the zenith lands in
        # the middle of the dome
        chain += (f"v360=e:fisheye:h_fov=180:v_fov=180:pitch={cfg.dome_pitch}"
                  f":w={cfg.dome_size}:h={cfg.dome_size},")
    chain += "format=yuv420p[v]"

    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *background,
            "-f", "rawvideo", "-pixel_format", "rgba",
            "-video_size", f"{cfg.width}x{cfg.height}",
            "-framerate", str(cfg.fps), "-i", "pipe:0",
            "-i", str(audio),
            "-filter_complex", chain,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-c:a", "aac", "-b:a", "320k",
            "-r", str(cfg.fps), "-shortest", str(out_path)]


def video_targets(cfg):
    """The videos `cfg` asks for, as the `(path, dome)` pairs `write_videos`
    takes.

    Returns:
      targets (:obj:`list`): one pair per video, panorama first.
    """
    stem = cfg.outdir / "stars_appearing"
    panorama = (Path(f"{stem}_panorama.mp4"), False)
    dome = (Path(f"{stem}_dome.mp4"), True)

    wanted = {"panorama": [panorama], "dome": [dome],
              "both": [panorama, dome]}
    if cfg.output not in wanted:
        raise ValueError(f"'{cfg.output}' is not an output. Choose from "
                         f"{sorted(wanted)}.")

    return wanted[cfg.output]


def write_videos(cfg, events, audio, targets=None, sky=None):
    """Render the frames once, into one ffmpeg process per output.

    Asked for both the panorama and the dome master, they are the same
    pixels reprojected differently, so they share a single pass of the
    frame generator rather than drawing everything twice.

    Args:
      targets (:obj:`list`, optional): `(path, dome)` pairs, one per
        output. Left out, `cfg.output` decides.
      sky (:obj:`pathlib.Path`, optional): the background panorama. Left
        out, `cfg.background` decides, rendering one if it says "auto".

    Returns:
      paths (:obj:`list`): the paths written, in the order given.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on your PATH.")

    if targets is None:
        targets = video_targets(cfg)

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    if sky is None:
        sky = resolve_background(cfg)

    n_frames = int(round(cfg.duration * cfg.fps))
    procs = [subprocess.Popen(encode(cfg, audio, path, sky=sky, dome=dome),
                              stdin=subprocess.PIPE)
             for path, dome in targets]

    label = ", ".join(path.name for path, _ in targets)
    try:
        for frame in tqdm.tqdm(render_frames(events, cfg), total=n_frames,
                               desc=label, unit="frame"):
            for proc in procs:
                proc.stdin.write(frame)
    finally:
        for proc in procs:
            proc.stdin.close()
        failed = [path for (path, _), proc in zip(targets, procs)
                  if proc.wait() != 0]

    if failed:
        raise RuntimeError("ffmpeg failed while writing "
                           f"{', '.join(str(p) for p in failed)}")

    return [path for path, _ in targets]


def write_video(cfg, events, audio, out_path, dome=False, sky=None):
    """Render every frame into a single ffmpeg process, ignoring `cfg.output`."""
    return write_videos(cfg, events, audio, [(out_path, dome)], sky=sky)[0]


# <u> __The whole sequence:__ </u>
#
# Everything above, in the order it has to happen, for the times you want the
# finished thing rather than a look at how it is made. `StarsAppearingLocal.ipynb`
# is the same run with each step in the open; `StarsAppearingColab.ipynb` is this
# one call behind a form.

@dataclass
class Sequence:
    """What one run of `make_sequence` produced."""

    cfg: Config
    sky: pd.DataFrame           # the stars, as `observed_sky` returned them
    frame: pd.DataFrame         # what was sonified
    events: pd.DataFrame        # what sounded, and when
    style: str                  # the style it was sonified with
    background: Path | None     # the panorama the stars were drawn over
    audio: Path                 # the rendered sonification
    videos: list                # the videos written, panorama first


def make_sequence(cfg, sound="Night Harp"):
    """Sky to finished video, in one call.

    Args:
      cfg (:obj:`Config`): every setting for the run.
      sound (`optional`, :obj:`str`): one of `SOUNDS`.

    Returns:
      result (:obj:`Sequence`): the outputs, and the tables behind them.
    """
    import strauss

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    sky = observed_sky(cfg)
    background = resolve_background(cfg)

    frame = star_frame(sky, cfg)
    style = chosen_style(sound, cfg)

    strauss.sonify(frame, style=style, channels=cfg.system,
                   duration=cfg.duration, angle_unit="degrees",
                   source_names=list(frame.index))
    events = sonified_events(frame)

    audio = cfg.outdir / "stars_appearing.wav"
    strauss.save(str(audio))

    # a re-run should start clean rather than adding a second sonification
    # alongside the first
    strauss.close()

    videos = write_videos(cfg, events, audio, sky=background)

    return Sequence(cfg=cfg, sky=sky, frame=frame, events=events, style=style,
                    background=background, audio=audio, videos=videos)


def show_videos(paths):
    """Play the finished videos in the notebook.

    The file is embedded rather than linked, since a notebook served from
    somewhere other than the working directory - `Colab`, say - cannot
    reach it by path.
    """
    from IPython.display import Video, display

    for path in paths:
        print(f"{Path(path).stat().st_size / 1e6:8.1f} MB  {path}")
        display(Video(str(path), embed=True))
