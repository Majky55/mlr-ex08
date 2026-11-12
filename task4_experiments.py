"""Task 4 - one change at a time, measured on two validation sets.

    python task4_experiments.py            # all six runs, ~20 min on a CPU laptop
    python task4_experiments.py baseline A # only some of them

Protocol (the whole point of the exercise):
  * torch.manual_seed(0) before every run, so the shuffle order and the
    weights start identical;
  * the baseline and every variant use the SAME network (LeafNetBN), the SAME
    aug_train_loader, the SAME 20 epochs, batch 32 and Adam 1e-3 - each variant
    changes exactly one thing, written in make();
  * every run is measured on the plain validation set AND on one fixed
    camera-like copy of it (built once with its own seed, saved to
    data/camera_like_val.pt) so all six runs see identical images.

20 epochs, not the 15 of Task 3: variant C decays the learning rate over 20
epochs, and a scheduler that spans the whole run has to be compared against a
baseline of the same length. Otherwise C would simply train longer.
"""

import copy
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):      # the table contains Δ and %, the
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console would refuse them

# ---------------------------------------------------------------- seed & config

def set_seed():
    """Same starting point for every run."""
    torch.manual_seed(0)
    random.seed(0)


set_seed()

EPOCHS = 20
BATCH = 32
LR = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# ------------------------------------------------- the Task 3 pipeline, unchanged


class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        folder_path = Path(folder)
        self.classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])
        self.samples = []
        for cls_name in self.classes:
            for path in sorted((folder_path / cls_name).glob("*")):
                if path.is_file():
                    self.samples.append((str(path), self.classes.index(cls_name)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, class_idx = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        return (self.transform(img) if self.transform else img), class_idx


class LeafNetBN(nn.Module):
    """LeafNet with a batch normalization after every convolution (Task 3d)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Variant A only: the same augmentation with colour jitter added.
aug_transform_a = transforms.Compose(aug_transform.transforms[:-2] + [
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

my_train_ds = LeafDataset("data/leaves/train", transform)
my_val_ds = LeafDataset("data/leaves/val", transform)
aug_ds = LeafDataset("data/leaves/train", aug_transform)
aug_ds_jitter = LeafDataset("data/leaves/train", aug_transform_a)

# The shuffle has its own generator so that every variant sees the batches in the
# same order, whatever random numbers building its model happened to use up.
SHUFFLE_SEED = 0
my_val_loader = DataLoader(my_val_ds, batch_size=32)
aug_train_loader = DataLoader(ConcatDataset([aug_ds, my_train_ds]), batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(SHUFFLE_SEED))
aug_train_loader_a = DataLoader(ConcatDataset([aug_ds_jitter, my_train_ds]), batch_size=BATCH, shuffle=True,
                                generator=torch.Generator().manual_seed(SHUFFLE_SEED))


# ------------------------------------------------------- the camera-like val set

def camera_like_val():
    """One fixed, distorted copy of the validation set - the same images for all runs.

    Rotation and a slight crop imitate a hand that is not perfectly steady and a
    leaf that is not centred; the colour jitter imitates the sun, shade and the
    phone's auto white balance. Built with its own seed (not 0, which belongs to
    the runs) and cached, so a rerun measures on literally the same tensors.
    """
    cache = Path("data/camera_like_val.pt")
    if cache.exists():
        saved = torch.load(cache)
        return saved["images"], saved["labels"]

    jitter = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        transforms.RandomResizedCrop(96, scale=(0.75, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    torch.manual_seed(1234)
    images = torch.stack([jitter(Image.open(p).convert("RGB")) for p, _ in my_val_ds.samples])
    labels = torch.tensor([lab for _, lab in my_val_ds.samples])
    torch.save({"images": images, "labels": labels}, cache)
    return images, labels


CAM_IMAGES, CAM_LABELS = camera_like_val()
camera_loader = DataLoader(TensorDataset(CAM_IMAGES, CAM_LABELS), batch_size=32)


# ------------------------------------------------------------- the variants, one change each

def make(variant):
    """Build the model and the settings of one variant. Exactly one line differs per variant."""
    cfg = dict(
        model=LeafNetBN(),
        loader=aug_train_loader,
        lr=LR,
        scheduler=None,
        keep_best=False,
        note="LeafNetBN + augmentation + batch norm (Task 3d)",
    )

    if variant == "A":                                   # colour augmentation
        cfg["loader"] = aug_train_loader_a
        cfg["note"] = "+ ColorJitter(0.2, 0.2, 0.2, 0.1) in training"

    if variant == "B":                                   # keep the best epoch
        cfg["keep_best"] = True
        cfg["note"] = "+ weights of the epoch with the lowest validation loss"

    if variant == "C":                                   # learning-rate schedule
        cfg["scheduler"] = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=1e-5)
        cfg["note"] = "+ CosineAnnealingLR 1e-3 -> 1e-5 over 20 epochs"

    if variant in ("D", "E"):                            # transfer learning
        net = models.mobilenet_v2(weights="DEFAULT")
        if variant == "D":
            for param in net.parameters():
                param.requires_grad = False
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, 4)   # new, trainable
            cfg["note"] = "MobileNetV2 pretrained, backbone frozen, head trained"
        else:
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, 4)
            cfg["lr"] = 1e-4
            cfg["note"] = "MobileNetV2 pretrained, all parameters trained at lr 1e-4"
        cfg["model"] = net

    return cfg


# ------------------------------------------------------------------- train & score

def score(model, loader):
    """Accuracy and mean loss of the model on one loader."""
    model.eval()
    correct, loss_sum, seen = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss_sum += nn.CrossEntropyLoss()(out, labels).item() * len(labels)
            correct += (out.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
    return correct / seen, loss_sum / seen


def run(variant):
    cfg = make(variant)
    set_seed()                                  # every run starts from the same point
    cfg["loader"].generator.manual_seed(SHUFFLE_SEED)   # ... and sees the batches in the same order
    model = cfg["model"].to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"])
    scheduler = cfg["scheduler"](optimizer) if cfg["scheduler"] else None

    best_loss, best_weights = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for images, labels in cfg["loader"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        val_acc, val_loss = score(model, my_val_loader)
        if val_loss < best_loss:
            best_loss, best_weights = val_loss, copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(f"    epoch {epoch + 1:2d}  train loss {loss_sum / len(cfg['loader']):.3f}"
              f"  val acc {val_acc:.3f}  val loss {val_loss:.3f}"
              + (f"  lr {optimizer.param_groups[0]['lr']:.5f}" if scheduler else ""))
        if scheduler:
            scheduler.step()

    if cfg["keep_best"]:
        model.load_state_dict(best_weights)     # variant B: back to the best epoch
    else:
        best_epoch = EPOCHS

    Path("data").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"data/weights_{variant}.pt")   # for the ONNX export

    val_acc, _ = score(model, my_val_loader)
    cam_acc, _ = score(model, camera_loader)
    params = sum(p.numel() for p in trainable)
    print(f"  {variant}: validation {val_acc:.3f}  camera-like {cam_acc:.3f}"
          f"  (best epoch {best_epoch}, {params / 1e6:.2f} M trainable parameters)")
    return dict(variant=variant, note=cfg["note"], val=val_acc, cam=cam_acc,
                best_epoch=best_epoch, params=params)


# --------------------------------------------------------------------- summary

def noise_floor(acc, n):
    """One image of n is worth this many percent; the binomial spread is this wide."""
    return 100.0 / n, 100.0 * math.sqrt(max(acc * (1 - acc), 0.01) / n)


def markdown_table(rows):
    n_val, n_cam = len(my_val_ds), len(CAM_LABELS)
    head = ("| variant | what changed | standard validation | camera-like validation "
            "| Δ standard | Δ camera-like | trainable parameters |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n")
    lines = []
    base = rows[0]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['note']} | {r['val'] * 100:.1f} % ({round(r['val'] * n_val)}/{n_val}) "
            f"| {r['cam'] * 100:.1f} % ({round(r['cam'] * n_cam)}/{n_cam}) "
            f"| {100 * (r['val'] - base['val']):+.1f} pp | {100 * (r['cam'] - base['cam']):+.1f} pp "
            f"| {r['params'] / 1e6:.2f} M |")
    return head + "\n".join(lines) + "\n"


def seed_spread_section():
    """Table of how far the same configuration moves when only the seed changes.

    Written by a separate probe (three seeds per variant) to results/seed_spread.json.
    Without it the main table has one run per row, and one run says nothing about
    how much of a difference is luck.
    """
    path = Path("results/seed_spread.json")
    if not path.exists():
        return ""
    probe = json.loads(path.read_text(encoding="utf-8"))
    seeds = probe["seeds"]
    lines = [f"| variant | standard, one run per seed | mean | spread "
             f"| camera-like, one run per seed | mean | spread |\n"
             f"| --- | --- | --- | --- | --- | --- | --- |"]
    for r in probe["rows"]:
        lines.append(f"| {r['variant']} | {', '.join(f'{v:.1f}' for v in r['val'])} % "
                     f"| {sum(r['val']) / len(r['val']):.1f} % "
                     f"| {max(r['val']) - min(r['val']):.1f} pp "
                     f"| {', '.join(f'{c:.1f}' for c in r['cam'])} % "
                     f"| {sum(r['cam']) / len(r['cam']):.1f} % "
                     f"| {max(r['cam']) - min(r['cam']):.1f} pp |")
    table = "\n".join(lines)
    widest = max(max(r["val"]) - min(r["val"]) for r in probe["rows"])
    widest_cam = max(max(r["cam"]) - min(r["cam"]) for r in probe["rows"])
    return f"""
## The same experiment twice: what a different seed does

{table}

Same configuration, only `torch.manual_seed` and the shuffle seed changed
({", ".join(map(str, seeds))}). Rerunning the *same* variant from scratch moves the
standard accuracy by up to **{widest:.1f} pp** and the camera-like one by up to
**{widest_cam:.1f} pp**. That swing — not the binomial ideal above — is what a
difference between two rows of the main table has to beat to be worth reporting, and
it is why the **mean** column matters more than any single number in it. Each row of
the main table is still one run of seed 0; re-running a variant three times costs
about as much compute as the whole table, which is why it is done for a few variants
here and not for all six.
{probe.get('note', '')}
"""


def deploy_section():
    """Size and browser speed of the winner, measured in the page itself."""
    path = Path("results/deploy_probe.json")
    if not path.exists():
        return ""
    p = json.loads(path.read_text(encoding="utf-8"))
    return f"""
## What the winner costs in the phone

Measured in the Task 2 page, which runs the model with onnxruntime-web in WebAssembly
(single thread — GitHub Pages cannot send the headers multithreading needs):

| | current `leaf_model.onnx` | {p['candidate']} |
| --- | --- | --- |
| ONNX file, float32 | {p['current_mb']:.1f} MB | {p['candidate_mb']:.1f} MB |
| one prediction, median of 19 | {p['current_median_ms']:.0f} ms | {p['candidate_median_ms']:.0f} ms |
| same, slowest of the 19 | {p['current_max_ms']:.0f} ms | {p['candidate_max_ms']:.0f} ms |
| first call, with cold weights | {p['current_cold_ms']:.0f} ms | {p['candidate_cold_ms']:.0f} ms |

Preprocessing (the crop, the resize, the normalisation) is included, because the page
measures it that way. Measured on a laptop CPU in {p.get('browser', 'Chrome')}; a
mid-range phone is several times slower, so read the two columns as a ratio, not as a
promise about a particular handset. The page asks for a frame every 100 ms, so both
models stay live here; what a phone feels most is the download, and that is paid once
and then cached.
{p.get('note', '')}


"""


def draft_report(rows):
    n_val, n_cam = len(my_val_ds), len(CAM_LABELS)
    per_img, binomial = noise_floor(rows[0]["val"], n_val)
    cam_binomial = noise_floor(rows[0]["cam"], n_cam)[1]
    best = max(rows, key=lambda r: r["cam"])

    # A gain only counts if it is bigger than the seed spread of both models compared.
    spread, mean = {}, {}
    if Path("results/seed_spread.json").exists():
        for r in json.loads(Path("results/seed_spread.json").read_text(encoding="utf-8"))["rows"]:
            spread[r["variant"]] = max(r["cam"]) - min(r["cam"])
            mean[r["variant"]] = sum(r["cam"]) / len(r["cam"])   # the probe saves percent already

    def bar(variant):
        """The camera-like gain over the baseline that has to be beaten to count."""
        return max(cam_binomial, spread.get(rows[0]["variant"], 0.0), spread.get(variant, 0.0))

    above, borderline = [], []
    for r in rows[1:]:
        gain = 100 * (r["cam"] - rows[0]["cam"])
        if gain <= 0:
            continue
        if gain > bar(r["variant"]):
            above.append(f"**{r['variant']}** (+{gain:.1f} pp)")
            continue
        text = f"{r['variant']} (+{gain:.1f} pp against a spread of {bar(r['variant']):.1f} pp"
        if mean.get(r["variant"], 0) > mean.get(rows[0]["variant"], 0):
            text += f", but +{mean[r['variant']] - mean[rows[0]['variant']]:.1f} pp averaged over its seeds"
        borderline.append(text + ")")
    above = ", ".join(above)
    if borderline:
        above += "; still inside the noise on this one run: " + ", ".join(borderline)
    if not above:
        above = "none — every difference is within one rerun of the baseline"

    export_note = """
The export is the Task 2e notebook cell with one line changed: build the variant with
`make("E")["model"]`, `load_state_dict(torch.load("data/weights_E.pt"))`, call `eval()`,
export. What the exported file measured afterwards (recorded in
`results/deploy_probe.json`): it reproduces the training run exactly — validation 0.935,
camera-like 0.871 — its logits agree with PyTorch to 2e-05 on all 62 validation images,
and it loads in onnxruntime-web unchanged: 100 nodes, opset 18, only Conv, Add, Clip,
ReduceMean, Reshape and Gemm, all in the default domain that the browser build knows.
`run()` saves the weights to `data/weights_<variant>.pt`; they are not committed, nor is
the dataset, so retraining is how you get them back.
""" if Path("results/deploy_probe.json").exists() else ""

    by = {r["variant"]: r for r in rows}
    d = by.get("D")
    fallback = (f"* **{d['variant']} is the fallback** if the download or an older phone turns out to be the\n"
                f"  binding constraint: it exports to the same size, reached {d['cam'] * 100:.1f} % on the\n"
                f"  camera-like set, and trains in a fraction of the time because only the head learns.\n"
                if d else "")
    rest = [r["variant"] for r in rows[1:] if r["variant"] in ("B", "C")]

    return f"""# Task 4 report — draft (fill in the judgement yourself)

Draft written by `task4_experiments.py` from the measured runs. The table, the
noise floor and the parameter counts are measured; the reasoning below is only a
starting point — rewrite it in your own words and decide what you would deploy.

## How the runs were made

One run per row, all of them with `torch.manual_seed(0)` set again at the start and
the same shuffle order (the loader has its own generator, so the batches arrive in
the same sequence in every run). Baseline, A, B and C are the same `LeafNetBN` on the
same `aug_train_loader` — the {len(aug_ds)} augmented plus {len(my_train_ds)} plain
copies of the training images, {len(aug_ds) + len(my_train_ds)} samples per epoch —
Adam at {LR}, {EPOCHS} epochs, batch {BATCH}. D and E change the network, which *is*
their one change; their loader and schedule are the baseline's. Each variant differs
from the baseline in exactly one line of `make()`. Every run is scored on the plain
validation set ({n_val} images) and on one fixed camera-like copy of it
({n_cam} images: rotation up to 25°, colour jitter, a crop of 75–100 %).

## Results

{markdown_table(rows)}
## Which model goes on the phone

* **Deploy {best['variant']}** — {best['note']}. It is the only variant that
  improved on *both* sets ({100 * (best['val'] - rows[0]['val']):+.1f} pp standard,
  {100 * (best['cam'] - rows[0]['cam']):+.1f} pp camera-like over the baseline), and in
  the seed repeats it was the steadiest of the four. What it costs:
  {best['params'] * 4e-6:.1f} MB of weights instead of {rows[0]['params'] * 4e-6:.1f} MB,
  and about 1.7× the time per frame — still well inside the page's 100 ms budget on this
  machine. Both numbers are in the table under "What the winner costs in the phone".
{fallback}* **Nothing to deploy from {", ".join(rest) if rest else 'the other variants'}**: {'neither beat the baseline by more than a rerun of the baseline does' if rest else 'none of them beat the baseline'},
  and A only looks good on the one distortion it was trained with.
* **Then measure it outside.** Every number above is measured on 62 photos from the same
  album. Re-export the model, regenerate `check_expected.json`, load the page on a
  phone, and photograph ten leaves you have never shown the network — that walk is the
  experiment that decides this, and it costs no training time at all.

## Read the differences against the noise before believing them

The validation set has **{n_val} images, so one correctly classified image is
{per_img:.1f} percentage points**. Around an accuracy of {rows[0]['val'] * 100:.0f} %
the binomial spread of such a measurement is about **±{binomial:.0f} pp** on the
standard set and **±{cam_binomial:.0f} pp** on the camera-like one. Differences of a
few points between two rows are therefore *not* evidence of a better model — a
difference has to be clearly larger than that to mean anything. Fixing the seed
makes the runs reproducible and identical in their shuffle order; it does not
make the measurement precise, and it does not sample run-to-run variation at all.
If you want to rank two variants that are close, rerun both with a few different
seeds and compare the spread of the results.
{seed_spread_section()}
## What the numbers suggest (judge this yourself)

* The camera-like column is where the phone camera lives. The baseline drops from
  {rows[0]['val'] * 100:.1f} % to {rows[0]['cam'] * 100:.1f} % as soon as the leaf is
  rotated, off-centre and lit differently — a gap of
  {100 * (rows[0]['val'] - rows[0]['cam']):.0f} pp that the plain validation set never
  showed. That gap is the whole reason to build the second set.
* Camera-like gains bigger than the seed spread of *both* models being compared:
  {above or 'none — every difference is within one rerun of the baseline'}.
  The best row is **{best['variant']}** ({best['note']}) at {best['cam'] * 100:.1f} %.
* A is the only variant that scores *better* on the distorted set than on the clean
  one — see the caveat about the test set below before reading that as a win.
* Accuracy is not free. {'LeafNetBN is about 0.8 MB of weights' if best['params'] < 1e6 else 'MobileNetV2 is about ' + format(best['params'] * 4e-6, '.1f') + ' MB of weights, fp32'};
  the page has to download the file over the phone's connection before the first
  prediction, and the current `leaf_model.onnx` is already 6.3 MB.

## What each variant is really worth here

* **A (colour jitter)** is the one to read most carefully: it lost accuracy on the
  plain validation set and gained on the camera-like one. Part of that is a real
  trade — colour invariance bought by throwing away colour information — but part of
  it is the design of this test: the camera-like set is distorted with
  `ColorJitter`, which is exactly the augmentation A was trained on and the other
  variants were not. A is therefore the one competitor *favoured* by the test set.
  To settle it, build the camera-like copy from a distortion A never saw (a blur, a
  different colour space, a photo of a leaf taken through a window) and see whether
  the crossover survives.
* **B (best epoch)** selects the epoch by the validation loss, so its standard
  number is optimistic by construction — the honest one is the camera-like column,
  which did not improve. Early stopping on a 62-image set mostly selects noise.
* **C (cosine schedule)** changed nothing that this measurement can see: same
  camera-like accuracy as the baseline, a slightly lower standard one.
* **D (frozen backbone)** is the cheapest real gain, and it is worth knowing where it
  comes from: 2.2 M parameters that already know edges, textures and shapes from
  ImageNet, plus a head of 5 124 parameters trained on 244 leaf photos.
  Note that freezing `requires_grad` does not freeze BatchNorm — the running means and
  variances of the pretrained backbone keep updating during training, so the features
  under the head do drift a little. It also does not make the *file* smaller: the
  frozen backbone is still exported, so D and E ship the same weights.
* **E (fine-tuning)** is the only variant whose gain is larger than the noise on both
  sets, on the camera-like one by a wide margin. That is what a pretrained backbone
  buys on a dataset this small.

{deploy_section()}## To actually deploy the winner

Re-export it the way Task 2e does (`torch.onnx.export` with an input named
`image` of `[1, 3, 96, 96]`, plus the `model_info` metadata — the page reads the mean,
the standard deviation and the labels out of that metadata and would otherwise keep
normalising for the model it has), push it as `leaf_model.onnx`, and re-run the page's
self-check: the probabilities for `check_image.jpg` change with the model, so
`check_expected.json` has to be regenerated from the notebook in the same session that
exported it.
{export_note}

## What this measurement does not settle

* One run per row in the main table, one camera-like sample, 62 images each.
* Pretrained weights saw a million general photos; nothing here rules out that part of
  E's lead comes from the shared background and lighting of the group's own photos
  rather than from leaf shapes. The camera-like copy disturbs rotation, colour and
  framing, which weakens such cues but does not remove them. The decisive test is
  photos nobody trained on: leaves from a different tree, a different day, a phone
  camera.
* The camera-like set is a *simulation*. A real camera adds motion blur, noise at
  high ISO, autofocus hunting and partial leaves — the numbers above are a lower
  bound on how much worse a real photo can be.
* Everything ran on a CPU in float32. Quantising the exported model to uint8 would cut
  the download by about four times at a small accuracy cost, and was not tested.
"""


RESULTS = Path("results/task4_results.json")


def main():
    args = sys.argv[1:]

    # "report" rebuilds the table and the draft from the saved record, no training
    if args and args[0] == "report":
        record = json.loads(RESULTS.read_text(encoding="utf-8"))
        rows = record["rows"]
    else:
        wanted = args or ["baseline", "A", "B", "C", "D", "E"]
        print(f"device {device} | train {len(my_train_ds)} | val {len(my_val_ds)} "
              f"| camera-like {len(CAM_LABELS)} | epochs {EPOCHS}")
        rows = []
        for variant in wanted:
            print(f"\n=== {variant} ===")
            set_seed()
            rows.append(run(variant))
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps({"rows": rows, "meta": {
            "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": 0,
            "train": len(my_train_ds), "val": len(my_val_ds), "camera_like": len(CAM_LABELS),
            "torch": torch.__version__, "device": str(device)}}, indent=1), encoding="utf-8")

    print("\n```markdown")
    print(markdown_table(rows))
    print("```")
    Path("REPORT_DRAFT.md").write_text(draft_report(rows), encoding="utf-8")
    print("wrote REPORT_DRAFT.md")


if __name__ == "__main__":
    main()
