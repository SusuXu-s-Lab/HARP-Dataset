"""Blue-tarp damage detection from pre- and post-event NAIP imagery.

Reference implementation of the detector behind the ``Ida_AerialDamaged`` and
``Ian_AerialDamaged`` columns of HARP.

After a hurricane, damaged roofs are commonly covered with blue plastic tarps.
A pixel is flagged as tarp when three conditions hold:

1. it is noticeably bluer than its red and green channels in the post image;
2. the same location became bluer between the pre and post images, which rules
   out pools and other surfaces that were already blue;
3. it is bright enough in the post image, which rules out dark tree shadows.

Pixel detections are aggregated to OpenStreetMap building footprints and then to
parcels. The thresholds were tuned by grid search against manually annotated
footprints (``tune_thresholds``). The two storms differ in how footprints become
parcel labels; see ``PARAMS`` and the two aggregation sections below.

Imagery, footprints, parcel records, and the manual annotations are not
redistributed. Running this file executes a self-contained example on synthetic
arrays.
"""

import itertools

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, morphology

# ── Parameters used for the released layers ─────────────────────────────────
PARAMS = {
    # 0.4 m NAIP. Tuned on 200 annotated tiles: P = 86.9 %, R = 90.0 %, F1 = 88.4 %.
    "ida": dict(
        thr_base=20,          # min increase in blueness, post minus pre
        b_post_min=20,        # min blueness in the post image
        bright_min=140,       # min brightness (max RGB channel) in the post image
        min_blob_px=150,      # drop tarp regions smaller than this (~24 m^2 at 0.4 m)
        min_footprint_px=20,  # skip footprints too small to judge
        frac_cutoff=0.05,     # footprint damaged if >= 5 % of its pixels are tarp
    ),
    # 1 m NAIP. The grid search on 57 annotated tiles selected bright_min = 170;
    # the released layer lowers it to 140 to recover shaded tarps.
    "ian": dict(
        thr_base=15,
        b_post_min=20,
        bright_min=140,
        min_blob_px=4,        # ~4 m^2 at 1 m
    ),
}

# Grids searched by tune_thresholds (frac_cutoff is the footprint-level cutoff).
GRIDS = {
    "ida": dict(thr_base=[-50, -30, -10, 0, 10, 20], b_post_min=[20, 40, 60, 80],
                bright_min=[40, 60, 80, 100, 120, 140],
                frac_cutoff=[0.01, 0.02, 0.05, 0.10, 0.15]),
    "ian": dict(thr_base=[5, 10, 15, 20, 25, 30], b_post_min=[15, 20, 25, 30],
                bright_min=[120, 130, 140, 150, 160, 170],
                frac_cutoff=[0.07, 0.08, 0.09, 0.10, 0.12, 0.15]),
}


# ── Pixel-level detector ────────────────────────────────────────────────────
def blueness(img):
    """Blue channel minus the larger of red and green, per pixel."""
    img = img.astype(float)
    return img[..., 2] - np.maximum(img[..., 0], img[..., 1])


def _match_size(pre, post):
    if post.shape[:2] != pre.shape[:2]:
        post = np.array(Image.fromarray(post).resize(
            (pre.shape[1], pre.shape[0]), Image.BILINEAR))
    return post


def detect_tarp(pre, post, p):
    """Boolean tarp mask for a co-registered (H, W, 3) uint8 RGB image pair."""
    post = _match_size(pre, post)
    valid = post.sum(axis=2) > 0                      # all-zero pixels are no-data
    b_post = blueness(post)
    mask = (valid
            & (b_post - blueness(pre) > p["thr_base"])   # condition 2
            & (b_post > p["b_post_min"])                 # condition 1
            & (post.max(axis=2) > p["bright_min"]))      # condition 3
    return morphology.remove_small_objects(mask, min_size=p["min_blob_px"])


# ── Geometry helpers ────────────────────────────────────────────────────────
def lonlat_to_px(lons, lats, bbox, shape):
    """Map lon/lat to pixel (x, y) inside a tile with bbox (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    H, W = shape
    xs = (np.asarray(lons, float) - min_lon) / (max_lon - min_lon) * W
    ys = (max_lat - np.asarray(lats, float)) / (max_lat - min_lat) * H
    return list(zip(xs, ys))


def rasterize(polygons_px, shape):
    """Fill polygons given as lists of pixel (x, y) vertices into a boolean mask."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons_px:
        draw.polygon([tuple(v) for v in poly], fill=1)
    return np.array(img, dtype=bool)


def pixel_area_m2(bbox, shape):
    """Approximate ground area of one pixel of a lon/lat tile."""
    min_lon, min_lat, max_lon, max_lat = bbox
    m_per_deg_lat = 111_000
    m_per_deg_lon = 111_000 * np.cos(np.radians((min_lat + max_lat) / 2))
    return ((max_lon - min_lon) * m_per_deg_lon / shape[1]) * \
           ((max_lat - min_lat) * m_per_deg_lat / shape[0])


# ── Ida: footprint fraction -> parcel ───────────────────────────────────────
def footprint_damaged_ida(tarp, footprint, p=PARAMS["ida"]):
    """True/False for one footprint mask, or None when it is too small to judge."""
    n_pix = int(footprint.sum())
    if n_pix < p["min_footprint_px"]:
        return None
    return bool((tarp & footprint).sum() / n_pix >= p["frac_cutoff"])


def apply_manual_labels(predicted, manual):
    """On the annotated tiles the manual label replaces the detector's prediction.

    Both arguments map a (tile_id, footprint_id) key to a boolean label.
    """
    return {key: manual.get(key, label) for key, label in predicted.items()}


def match_parcels_to_footprints(parcels, footprints, snap_m=5.0):
    """Assign each parcel point to a footprint: containing polygon first, else nearest within snap_m.

    ``parcels`` and ``footprints`` are GeoDataFrames in EPSG:4326 with columns
    ``parcel_id`` and ``footprint_id``. A parcel then inherits the damage label
    of its footprint.
    """
    import geopandas as gpd

    within = (gpd.sjoin(parcels, footprints, predicate="within", how="left")
                 .drop(columns="index_right").drop_duplicates("parcel_id"))
    todo = within.footprint_id.isna()
    if todo.any():
        near = gpd.sjoin_nearest(parcels[parcels.parcel_id.isin(within.parcel_id[todo])].to_crs(3857),
                                 footprints.to_crs(3857), how="left", max_distance=snap_m)
        near = near.drop_duplicates("parcel_id").set_index("parcel_id").footprint_id
        within.loc[todo, "footprint_id"] = within.loc[todo, "parcel_id"].map(near)
    return within[["parcel_id", "footprint_id"]]


# ── Ian: parcel centroid -> building component -> tarp area ────────────────
def parcel_blue_area_ian(tarp, footprint_labels, col, row, area_m2):
    """Tarp area (m^2) on the building that contains the parcel's centroid pixel.

    ``footprint_labels`` is ``measure.label(all_footprints_mask, connectivity=2)``
    for the tile. A parcel is flagged when this area is greater than zero; when a
    parcel falls in several overlapping tiles the largest value is kept.
    """
    H, W = footprint_labels.shape
    if not (0 <= col < W and 0 <= row < H):
        return 0.0
    comp = footprint_labels[row, col]
    if comp == 0:
        return 0.0
    return float((tarp & (footprint_labels == comp)).sum()) * area_m2


# ── Threshold tuning against manual labels ─────────────────────────────────
def footprint_stats(pre, post, footprint, damaged):
    """Per-footprint pixel values needed to evaluate any threshold combination."""
    post = _match_size(pre, post)
    b_post = blueness(post)
    return dict(bd=(b_post - blueness(pre))[footprint], bpost=b_post[footprint],
                bright=post.max(axis=2)[footprint], valid=(post.sum(axis=2) > 0)[footprint],
                n_pix=int(footprint.sum()), damaged=bool(damaged))


def tune_thresholds(stats, grid):
    """Grid search that returns the F1-maximizing thresholds and their P/R/F1."""
    best = None
    for tb, bm, brm, fc in itertools.product(grid["thr_base"], grid["b_post_min"],
                                             grid["bright_min"], grid["frac_cutoff"]):
        tp = fp = fn = 0
        for s in stats:
            blue = s["valid"] & (s["bd"] > tb) & (s["bpost"] > bm) & (s["bright"] > brm)
            pred = blue.sum() / s["n_pix"] >= fc
            tp += pred and s["damaged"]
            fp += pred and not s["damaged"]
            fn += (not pred) and s["damaged"]
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        if best is None or F1 > best["F1"]:
            best = dict(thr_base=tb, b_post_min=bm, bright_min=brm, frac_cutoff=fc,
                        P=P, R=R, F1=F1)
    return best


# ── Example on synthetic arrays ────────────────────────────────────────────
def _synthetic_tile():
    """80x80 tile: a tarped roof, a roof under a new dark shadow, and a pool."""
    pre = np.full((80, 80, 3), 90, np.uint8)
    pre[10:34, 10:34] = (150, 150, 150)     # house A roof
    pre[10:34, 46:70] = (150, 150, 150)     # house B roof
    pre[50:66, 20:44] = (40, 120, 200)      # pool, blue before the storm
    post = pre.copy()
    post[14:30, 14:30] = (70, 120, 225)     # house A: bright blue tarp
    post[14:30, 50:66] = (25, 35, 95)       # house B: dark bluish shadow
    houses = {"A": [(10, 10), (33, 10), (33, 33), (10, 33)],
              "B": [(46, 10), (69, 10), (69, 33), (46, 33)]}
    return pre, post, houses


if __name__ == "__main__":
    pre, post, houses = _synthetic_tile()
    shape = pre.shape[:2]

    print("Pixel detector (Ida parameters)")
    tarp = detect_tarp(pre, post, PARAMS["ida"])
    print(f"  tarp pixels on house A roof : {int(tarp[10:34, 10:34].sum())}")
    print(f"  tarp pixels on house B roof : {int(tarp[10:34, 46:70].sum())}  (shadow fails brightness)")
    print(f"  tarp pixels on the pool     : {int(tarp[50:66, 20:44].sum())}  (no pre-to-post change)")

    print("\nIda rule: footprint fraction")
    for name, poly in houses.items():
        fp = rasterize([poly], shape)
        print(f"  house {name}: damaged = {footprint_damaged_ida(tarp, fp)}")

    print("\nIan rule: tarp area on the parcel's building")
    tarp_ian = detect_tarp(pre, post, PARAMS["ian"])
    labels = measure.label(rasterize(houses.values(), shape), connectivity=2)
    for name, (col, row) in {"A": (22, 22), "B": (58, 22)}.items():
        area = parcel_blue_area_ian(tarp_ian, labels, col, row, area_m2=1.0)
        print(f"  parcel on house {name}: tarp area = {area:.0f} m^2 -> damaged = {area > 0}")

    print("\nThreshold tuning on the two labeled houses")
    stats = [footprint_stats(pre, post, rasterize([houses["A"]], shape), True),
             footprint_stats(pre, post, rasterize([houses["B"]], shape), False)]
    small_grid = dict(thr_base=[0, 20], b_post_min=[20], bright_min=[40, 140], frac_cutoff=[0.05])
    print(f"  best: {tune_thresholds(stats, small_grid)}")
