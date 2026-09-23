"""Fine-tuning DamageCAT on FEMA damage labels for Hurricane Ida.

Reference implementation of the change-detection half of ``Ida_AerialDamaged``
(the other half is the blue-tarp detector in ``blue_tarp_detection.py``).

Pipeline:

1. Labels. Each FEMA Visual Damage Assessment point is snapped to the nearest
   OpenStreetMap building footprint within 5 m and the footprint takes its most
   severe class. Minor, Major, and Destroyed are damage; No Visible Damage is
   intact. Affected is excluded, and every pixel outside a labeled footprint is
   ignored by the loss.
2. Split. Tiles are split 80/20 with stratification by each tile's most severe
   class, and only tiles with at least one damaged footprint are kept.
3. Training. The pretrained DamageCAT checkpoint is fine-tuned on a binary target
   with Dice plus focal loss. Tiles holding the rarer Major and Destroyed
   footprints are oversampled.
4. Inference. Softmax outputs are averaged over four flips, and a footprint is
   damaged when at least 1 % of its pixels are predicted as damage.

The network definition and pretrained weights come from the upstream DamageCAT
repository (https://github.com/YimingXiao98/DamageCAT, Apache-2.0; weights at
https://zenodo.org/records/15454349). The fine-tuned weights, imagery, FEMA
points, and footprints are not redistributed. Running this file exercises the label, split, sampling, and
decision steps on synthetic inputs; training needs the upstream repository and
``segmentation_models_pytorch``.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, WeightedRandomSampler

# ── Configuration used for the released layer ──────────────────────────────
SEED = 42
N_CLASS = 5                 # DamageCAT head: 0 intact, 1 partial roof, 2 total roof,
                            # 3 partial collapse, 4 total collapse
IGNORE_INDEX = 255
IMG_SIZE = 512
SNAP_RADIUS_M = 5.0
TEST_FRAC = 0.2
MIN_DAMAGED_FOOTPRINTS = 1  # density filter per tile

BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
EVAL_EVERY = 2
DAMAGE_FRAC_CUTOFF = 0.01   # footprint damaged if >= 1 % of its pixels are damage

# FEMA level -> DamageCAT class. Affected is deliberately absent, so its
# footprints are never painted and stay at IGNORE_INDEX.
FEMA_TO_CLASS = {"No Visible Damage": 0, "Minor": 1, "Major": 2, "Destroyed": 4}

# Relative share of training draws for tiles whose most severe footprint is ...
# After the density filter every training tile holds a damaged footprint, so the
# "intact" share receives no tiles in practice.
TARGET_SHARES = {"destroyed": 0.20, "major": 0.25, "minor": 0.25, "intact": 0.30}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 1. Labels ──────────────────────────────────────────────────────────────
def snap_fema_to_footprints(fema, footprints):
    """Per-footprint label from FEMA points snapped within SNAP_RADIUS_M.

    ``fema`` is a GeoDataFrame (EPSG:4326) with a ``DamageLevel`` column;
    ``footprints`` has a ``footprint_id`` column. Returns footprint_id -> class.
    """
    import geopandas as gpd

    fema = fema[fema.DamageLevel.isin(FEMA_TO_CLASS)].copy()
    fema["cls"] = fema.DamageLevel.map(FEMA_TO_CLASS).astype(int)
    snap = gpd.sjoin_nearest(fema.to_crs(3857)[["cls", "geometry"]],
                             footprints.to_crs(3857)[["footprint_id", "geometry"]],
                             how="inner", max_distance=SNAP_RADIUS_M)
    return snap.groupby("footprint_id").cls.max().to_dict()


def rasterize_tile_mask(labeled_polygons_px, shape):
    """Training mask for one tile: labeled footprints painted, everything else ignored.

    ``labeled_polygons_px`` is a list of (pixel-vertex list, class). Milder
    classes are painted first so that overlaps keep the more severe class.
    """
    img = Image.new("L", (shape[1], shape[0]), IGNORE_INDEX)
    draw = ImageDraw.Draw(img)
    for poly, cls in sorted(labeled_polygons_px, key=lambda pc: pc[1]):
        draw.polygon([tuple(v) for v in poly], fill=int(cls))
    return np.array(img, dtype=np.uint8)


# ── 2. Split ───────────────────────────────────────────────────────────────
def split_tiles(tile_classes, seed=SEED, test_frac=TEST_FRAC):
    """Stratified train/test split of tiles, then the damage-density filter.

    ``tile_classes`` maps tile_id -> list of footprint classes on that tile.
    """
    max_cls = {t: max(c) for t, c in tile_classes.items()}
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in sorted(set(max_cls.values())):
        ids = sorted(t for t, m in max_cls.items() if m == cls)
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_frac))) if len(ids) > 1 else 0
        test += ids[:n_test]
        train += ids[n_test:]

    def dense(t):
        return sum(c > 0 for c in tile_classes[t]) >= MIN_DAMAGED_FOOTPRINTS
    return sorted(filter(dense, train)), sorted(filter(dense, test))


def tile_category(classes):
    if 4 in classes:
        return "destroyed"
    if 2 in classes:
        return "major"
    if 1 in classes:
        return "minor"
    return "intact"


# ── 3. Training ────────────────────────────────────────────────────────────
class FineTuneDataset(Dataset):
    """Reads ``images/pre_<id>.png``, ``images/post_<id>.png`` and ``masks/post_<id>.png``."""

    def __init__(self, split_dir, tile_ids, is_train):
        self.img_dir = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.tile_ids = list(tile_ids)
        self.is_train = is_train

    def __len__(self):
        return len(self.tile_ids)

    @staticmethod
    def _to_tensor(rgb):
        # DamageCAT was trained on BGR input normalized to [-1, 1].
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        return (torch.from_numpy(bgr).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]
        pre = np.array(Image.open(os.path.join(self.img_dir, f"pre_{tid}.png")).convert("RGB"))
        post = np.array(Image.open(os.path.join(self.img_dir, f"post_{tid}.png")).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, f"post_{tid}.png")))

        if self.is_train:
            # Binary target: every damage class becomes 1.
            mask = np.where((mask >= 1) & (mask <= 4), np.uint8(1), mask)

        pad_h, pad_w = IMG_SIZE - mask.shape[0], IMG_SIZE - mask.shape[1]
        if pad_h > 0 or pad_w > 0:     # edge tiles are smaller than 512 x 512
            pre = np.pad(pre, [(0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)])
            post = np.pad(post, [(0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)])
            mask = np.pad(mask, [(0, max(0, pad_h)), (0, max(0, pad_w))])

        if self.is_train:
            if random.random() < 0.5:
                pre, post, mask = pre[:, ::-1], post[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                pre, post, mask = pre[::-1], post[::-1], mask[::-1]

        return {"pre": self._to_tensor(pre), "post": self._to_tensor(post),
                "mask": torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64))}


def make_sampler(train_categories):
    """Class-aware sampler: each category's expected share of draws follows TARGET_SHARES."""
    counts = {c: 0 for c in TARGET_SHARES}
    for c in train_categories:
        counts[c] += 1
    total = sum(TARGET_SHARES.values())
    weight = {c: (TARGET_SHARES[c] / total) / max(counts[c], 1) for c in TARGET_SHARES}
    w = torch.tensor([weight[c] for c in train_categories], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(train_categories), replacement=True)


def load_damagecat(damagecat_dir, ckpt_path, device):
    """Build the upstream DamageCAT network and load a checkpoint."""
    sys.path.insert(0, damagecat_dir)
    from models.networks import define_G

    gpu_ids = [0] if device.type == "cuda" else []
    args = argparse.Namespace(gpu_ids=gpu_ids, net_G="newUNetTrans", n_class=N_CLASS,
                              img_size=IMG_SIZE, print_models=False)
    net = define_G(args=args, gpu_ids=gpu_ids)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_G_state_dict"])
    return net.to(device)


def make_criterion():
    """Dice plus focal loss, the combination DamageCAT itself trains with."""
    import segmentation_models_pytorch.losses as L

    dice = L.DiceLoss(mode="multiclass", from_logits=True, ignore_index=IGNORE_INDEX)
    focal = L.FocalLoss(mode="multiclass", alpha=0.5, gamma=2.0, ignore_index=IGNORE_INDEX)
    return lambda logits, target: dice(logits, target) + focal(logits, target)


@torch.no_grad()
def binary_f1(net, loader, device):
    """Mean per-tile pixel F1 of any-damage (class > 0) vs intact, ignoring unlabeled pixels."""
    net.eval()
    scores = []
    for b in loader:
        pred = net(b["pre"].to(device), b["post"].to(device)).argmax(1).cpu().numpy()
        gt = b["mask"].numpy()
        valid = gt != IGNORE_INDEX
        for g, p, v in zip(gt > 0, pred > 0, valid):
            g, p = g & v, p & v
            tp, fp, fn = (g & p).sum(), (~g & p).sum(), (g & ~p).sum()
            if tp + fp + fn:
                prec, rec = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
                scores.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(scores)) if scores else 0.0


def train(net, train_loader, test_loader, device, out_dir):
    """Fine-tune and keep both the best-F1 and the final checkpoint."""
    criterion = make_criterion()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=LR * 0.05)
    best = binary_f1(net, test_loader, device)    # the pretrained model is the bar to beat
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(net(b["pre"].to(device), b["post"].to(device)), b["mask"].to(device))
            loss.backward()
            opt.step()
        sched.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            f1 = binary_f1(net, test_loader, device)
            print(f"epoch {epoch:>2}  loss {loss.item():.4f}  binary F1 {f1:.3f}")
            if f1 > best:
                best = f1
                torch.save({"model_G_state_dict": net.state_dict(), "epoch": epoch},
                           os.path.join(out_dir, "best_ckpt.pt"))
    # The released predictions were produced with this final-epoch checkpoint.
    torch.save({"model_G_state_dict": net.state_dict(), "epoch": NUM_EPOCHS},
               os.path.join(out_dir, "last_ckpt.pt"))


# ── 4. Inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def predict_tta(net, pre, post, device):
    """Argmax class map after averaging softmax over identity, h-flip, v-flip, and both."""
    net.eval()
    a = FineTuneDataset._to_tensor(pre).unsqueeze(0).to(device)
    b = FineTuneDataset._to_tensor(post).unsqueeze(0).to(device)
    probs = 0
    for dims in ([], [-1], [-2], [-2, -1]):
        f = (lambda t: torch.flip(t, dims)) if dims else (lambda t: t)
        probs = probs + f(torch.softmax(net(f(a), f(b)), dim=1))
    return (probs / 4).argmax(1).squeeze(0).cpu().numpy()


def footprint_damaged(class_map, footprint):
    """A footprint is damaged when >= DAMAGE_FRAC_CUTOFF of its pixels are predicted as damage."""
    inside = class_map[footprint]
    return bool(inside.size and (inside > 0).mean() >= DAMAGE_FRAC_CUTOFF)


# ── Example on synthetic inputs ────────────────────────────────────────────
if __name__ == "__main__":
    seed_everything()

    print("1. Label mask for a 40x40 tile with three footprints")
    footprints = [([(2, 2), (12, 2), (12, 12), (2, 12)], FEMA_TO_CLASS["Minor"]),
                  ([(20, 2), (30, 2), (30, 12), (20, 12)], FEMA_TO_CLASS["No Visible Damage"]),
                  ([(2, 20), (12, 20), (12, 30), (2, 30)], FEMA_TO_CLASS["Destroyed"])]
    mask = rasterize_tile_mask(footprints, (40, 40))
    values, counts = np.unique(mask, return_counts=True)
    print(f"   pixel counts by value: {dict(zip(values.tolist(), counts.tolist()))}"
          f"  (255 = ignored; Affected would stay 255)")

    print("\n2. Stratified split of 20 made-up tiles")
    rng = np.random.default_rng(0)
    tiles = {t: rng.choice([0, 0, 1, 1, 1, 2, 4], size=rng.integers(1, 6)).tolist() for t in range(20)}
    tiles[99] = [0, 0]                        # intact-only tile, removed by the density filter
    train_ids, test_ids = split_tiles(tiles)
    print(f"   train {train_ids}\n   test  {test_ids}")

    print("\n3. Oversampling: share of 4,000 draws by tile category")
    cats = [tile_category(tiles[t]) for t in train_ids]
    draws = list(WeightedRandomSampler(make_sampler(cats).weights, 4000, replacement=True))
    for c in TARGET_SHARES:
        n_tiles = cats.count(c)
        share = sum(cats[i] == c for i in draws) / len(draws)
        print(f"   {c:<9} {n_tiles:>2} tiles  ->  {share:5.1%} of draws")

    print("\n4. Footprint decision on a synthetic class map")
    class_map = np.zeros((40, 40), np.uint8)
    class_map[3:5, 3:5] = 1                   # 4 damage pixels on the first footprint
    for poly, _ in footprints[:2]:
        fp = rasterize_tile_mask([(poly, 1)], (40, 40)) == 1
        frac = (class_map[fp] > 0).mean()
        print(f"   footprint with {frac:.1%} damage pixels -> damaged = {footprint_damaged(class_map, fp)}")
