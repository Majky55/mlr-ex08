# Task 4 report — draft (fill in the judgement yourself)

Draft written by `task4_experiments.py` from the measured runs. The table, the
noise floor and the parameter counts are measured; the reasoning below is only a
starting point — rewrite it in your own words and decide what you would deploy.

## How the runs were made

One run per row, all of them with `torch.manual_seed(0)` set again at the start and
the same shuffle order (the loader has its own generator, so the batches arrive in
the same sequence in every run). Baseline, A, B and C are the same `LeafNetBN` on the
same `aug_train_loader` — the 244 augmented plus 244 plain
copies of the training images, 488 samples per epoch —
Adam at 0.001, 20 epochs, batch 32. D and E change the network, which *is*
their one change; their loader and schedule are the baseline's. Each variant differs
from the baseline in exactly one line of `make()`. Every run is scored on the plain
validation set (62 images) and on one fixed camera-like copy of it
(62 images: rotation up to 25°, colour jitter, a crop of 75–100 %).

## Results

| variant | what changed | standard validation | camera-like validation | Δ standard | Δ camera-like | trainable parameters |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | LeafNetBN + augmentation + batch norm (Task 3d) | 79.0 % (49/62) | 54.8 % (34/62) | +0.0 pp | +0.0 pp | 0.21 M |
| A | + ColorJitter(0.2, 0.2, 0.2, 0.1) in training | 64.5 % (40/62) | 67.7 % (42/62) | -14.5 pp | +12.9 pp | 0.21 M |
| B | + weights of the epoch with the lowest validation loss | 77.4 % (48/62) | 51.6 % (32/62) | -1.6 pp | -3.2 pp | 0.21 M |
| C | + CosineAnnealingLR 1e-3 -> 1e-5 over 20 epochs | 74.2 % (46/62) | 54.8 % (34/62) | -4.8 pp | +0.0 pp | 0.21 M |
| D | MobileNetV2 pretrained, backbone frozen, head trained | 83.9 % (52/62) | 64.5 % (40/62) | +4.9 pp | +9.7 pp | 0.01 M |
| E | MobileNetV2 pretrained, all parameters trained at lr 1e-4 | 93.5 % (58/62) | 87.1 % (54/62) | +14.5 pp | +32.3 pp | 2.23 M |

## Which model goes on the phone

* **Deploy E** — MobileNetV2 pretrained, all parameters trained at lr 1e-4. It is the only variant that
  improved on *both* sets (+14.5 pp standard,
  +32.3 pp camera-like over the baseline), and in
  the seed repeats it was the steadiest of the four. What it costs:
  8.9 MB of weights instead of 0.8 MB,
  and about 1.7× the time per frame — still well inside the page's 100 ms budget on this
  machine. Both numbers are in the table under "What the winner costs in the phone".
* **D is the fallback** if the download or an older phone turns out to be the
  binding constraint: it exports to the same size, reached 64.5 % on the
  camera-like set, and trains in a fraction of the time because only the head learns.
* **Nothing to deploy from B, C**: neither beat the baseline by more than a rerun of the baseline does,
  and A only looks good on the one distortion it was trained with.
* **Then measure it outside.** Every number above is measured on 62 photos from the same
  album. Re-export the model, regenerate `check_expected.json`, load the page on a
  phone, and photograph ten leaves you have never shown the network — that walk is the
  experiment that decides this, and it costs no training time at all.

## Read the differences against the noise before believing them

The validation set has **62 images, so one correctly classified image is
1.6 percentage points**. Around an accuracy of 79 %
the binomial spread of such a measurement is about **±5 pp** on the
standard set and **±6 pp** on the camera-like one. Differences of a
few points between two rows are therefore *not* evidence of a better model — a
difference has to be clearly larger than that to mean anything. Fixing the seed
makes the runs reproducible and identical in their shuffle order; it does not
make the measurement precise, and it does not sample run-to-run variation at all.
If you want to rank two variants that are close, rerun both with a few different
seeds and compare the spread of the results.

## The same experiment twice: what a different seed does

| variant | standard, one run per seed | mean | spread | camera-like, one run per seed | mean | spread |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 79.0, 77.4, 75.8 % | 77.4 % | 3.2 pp | 54.8, 53.2, 48.4 % | 52.1 % | 6.4 pp |
| A | 64.5, 75.8, 72.6 % | 71.0 % | 11.3 pp | 67.7, 61.3, 66.1 % | 65.0 % | 6.4 pp |
| D | 83.9, 85.5, 82.3 % | 83.9 % | 3.2 pp | 64.5, 72.6, 75.8 % | 71.0 % | 11.3 pp |
| E | 93.5, 90.3, 93.5 % | 92.4 % | 3.2 pp | 87.1, 85.5, 87.1 % | 86.6 % | 1.6 pp |

Same configuration, only `torch.manual_seed` and the shuffle seed changed
(0, 1, 2). Rerunning the *same* variant from scratch moves the
standard accuracy by up to **11.3 pp** and the camera-like one by up to
**11.3 pp**. That swing — not the binomial ideal above — is what a
difference between two rows of the main table has to beat to be worth reporting, and
it is why the **mean** column matters more than any single number in it. Each row of
the main table is still one run of seed 0; re-running a variant three times costs
about as much compute as the whole table, which is why it is done for a few variants
here and not for all six.

Three things fall out of this that the main table hides.

* **Variant A is the least reproducible of the four on the standard set** (11.3 pp
  across three seeds), so its headline -14.5 pp is largely the seed it happened to get.
  By the mean of three runs, colour jitter costs 6.4 pp of standard accuracy and buys
  12.9 pp of camera-like accuracy - a smaller and more believable trade than the table's
  -14.5 / +12.9 suggests.
* **E is the most stable** (1.6 pp of spread on the camera-like set, a single image).
  Its lead is 20 times its own spread, and it holds on the means too
  (86.6 % vs the baseline's 52.1 %).
* **D's single seed-0 run is not yet evidence**: +9.7 pp over the baseline against a
  spread of 11.3 pp of its own. Averaged over three seeds it does look real
  (71.0 % vs 52.1 %, +18.9 pp), but it is also the variant whose camera-like accuracy
  moved most, so do not quote its seed-0 number alone.

Three seeds is still a small sample - these means have a spread of their own, roughly
the single-run spread divided by the square root of three.


## What the numbers suggest (judge this yourself)

* The camera-like column is where the phone camera lives. The baseline drops from
  79.0 % to 54.8 % as soon as the leaf is
  rotated, off-centre and lit differently — a gap of
  24 pp that the plain validation set never
  showed. That gap is the whole reason to build the second set.
* Camera-like gains bigger than the seed spread of *both* models being compared:
  **A** (+12.9 pp), **E** (+32.3 pp); still inside the noise on this one run: D (+9.7 pp against a spread of 11.3 pp, but +18.8 pp averaged over its seeds).
  The best row is **E** (MobileNetV2 pretrained, all parameters trained at lr 1e-4) at 87.1 %.
* A is the only variant that scores *better* on the distorted set than on the clean
  one — see the caveat about the test set below before reading that as a win.
* Accuracy is not free. MobileNetV2 is about 8.9 MB of weights, fp32;
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


## What the winner costs in the phone

Measured in the Task 2 page, which runs the model with onnxruntime-web in WebAssembly
(single thread — GitHub Pages cannot send the headers multithreading needs):

| | current `leaf_model.onnx` | variant E — fine-tuned MobileNetV2 |
| --- | --- | --- |
| ONNX file, float32 | 6.3 MB | 9.1 MB |
| one prediction, median of 19 | 18 ms | 32 ms |
| same, slowest of the 19 | 30 ms | 55 ms |
| first call, with cold weights | 83 ms | 118 ms |

Preprocessing (the crop, the resize, the normalisation) is included, because the page
measures it that way. Measured on a laptop CPU in Chrome (headless) + onnxruntime-web 1.31.1, WebAssembly, single thread; a
mid-range phone is several times slower, so read the two columns as a ratio, not as a
promise about a particular handset. The page asks for a frame every 100 ms, so both
models stay live here; what a phone feels most is the download, and that is paid once
and then cached.

Both files were timed in the same page, back to back, several times: the current model
came out at 18-29 ms and E at 32-59 ms depending on what else the laptop was doing, and
E was slower in every pair. The ratio, about 1.7x, is the number worth carrying to a
phone; the absolute milliseconds are this machine's. Both stay under the page's
100 ms-per-frame budget here, and both fit the browser's memory comfortably.



## To actually deploy the winner

Re-export it the way Task 2e does (`torch.onnx.export` with an input named
`image` of `[1, 3, 96, 96]`, plus the `model_info` metadata — the page reads the mean,
the standard deviation and the labels out of that metadata and would otherwise keep
normalising for the model it has), push it as `leaf_model.onnx`, and re-run the page's
self-check: the probabilities for `check_image.jpg` change with the model, so
`check_expected.json` has to be regenerated from the notebook in the same session that
exported it.

The export is the Task 2e notebook cell with one line changed: build the variant with
`make("E")["model"]`, `load_state_dict(torch.load("data/weights_E.pt"))`, call `eval()`,
export. What the exported file measured afterwards (recorded in
`results/deploy_probe.json`): it reproduces the training run exactly — validation 0.935,
camera-like 0.871 — its logits agree with PyTorch to 2e-05 on all 62 validation images,
and it loads in onnxruntime-web unchanged: 100 nodes, opset 18, only Conv, Add, Clip,
ReduceMean, Reshape and Gemm, all in the default domain that the browser build knows.
`run()` saves the weights to `data/weights_<variant>.pt`; they are not committed, nor is
the dataset, so retraining is how you get them back.


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
