# mlr-ex08 — leaf classifier in the browser

**Page: <https://majky55.github.io/mlr-ex08/>** — open it on a phone, press
*Start camera*, and the leaf classifier trained in `ex08_student.ipynb` runs in
the phone's own browser. Nothing is uploaded: the only network requests are for
this page, `leaf_model.onnx`, `model_info.json` and the `onnxruntime-web`
scripts from jsDelivr.

## Files

| File | What it is |
| --- | --- |
| `index.html` | The whole app: camera, preprocessing, inference, results, QR code |
| `leaf_model.onnx` | The exported model — input `image` `[1, 3, 96, 96]`, output `logits` `[1, 4]` |
| `model_info.json` | Classes `apple, cherry, chestnut, maple`, `input_size` 96, `mean`, `std` |
| `check_image.jpg` | Validation image with the notebook's probabilities |
| `check_expected.json` | Those probabilities, for the self-check |
| `task4_experiments.py` | Task 4: the five optimization runs, one change each, and the tables |
| `REPORT_DRAFT.md` | Draft the same script writes from its measured runs |
| `results/` | The measured records behind those tables (runs, seed repeats, phone costs) |

## How the frame is preprocessed

Per frame, in `toTensor()` in `index.html`:

1. crop the **central square** of the camera frame,
2. resize it to **96 × 96** (`input_size` from `model_info.json`),
3. divide the RGB bytes by 255 (`ToTensor`),
4. standardize each channel with `(x - mean) / std` using the `mean`/`std` from
   `model_info.json` (`Normalize`),
5. fill a `Float32Array` channel-first, as `[1, 3, 96, 96]`.

The model's last layer is `logits`, not probabilities, so the page applies
softmax itself — the same `model(x).softmax(dim=1)` as the notebook.

## Checking it

* **Same numbers as the notebook:** press *Run self-check* (or upload
  `check_image.jpg`). The page prints both columns and PASS/FAIL at three
  decimals. Measured in Chrome: largest difference **7.5e-7**, e.g. apple
  `0.936299` here vs `0.936298` in `check_expected.json`.
* **Live camera:** the square overlay is exactly the region the model sees;
  all four class probabilities update about ten times a second.
* **QR code:** generated in the page by its own encoder (`buildQrMatrix`, byte
  mode, error level M) — no QR web service is involved. It encodes the URL
  above and is printed next to it so you can read the address too.

Needs HTTPS (GitHub Pages provides it) and a browser that allows `getUserMedia`;
the camera starts on the button press because iOS Safari only lets a gesture
open it. Inference is plain single-threaded WebAssembly, so no COOP/COEP headers
are needed — which GitHub Pages cannot send anyway.

It was verified here in headless Chrome (self-check PASS, live camera on a fake
device). **Safari on iOS has not been tested from this machine** — open it on
your own phone and write the result, with a screenshot, into your `REVIEW.md`.

## Task 4 — optimizing the classifier for the camera

    pip install torch torchvision pillow
    python task4_experiments.py          # ~20 min on a CPU laptop, no GPU needed

One run per variant, `torch.manual_seed(0)` and the same batch order every time,
the training images of `data/leaves/{train,val}` (the group's photos — they stay
out of this repository), and every run scored twice: on the validation set and on
one fixed **camera-like** copy of it (rotation, colour jitter, a slight crop, built
once with its own seed). The model in the page today is the baseline row.

| variant | standard validation | camera-like |
| --- | --- | --- |
| baseline — `LeafNetBN` from Task 3 | 79.0 % | 54.8 % |
| A color jitter | 64.5 % | 67.7 % |
| B best-epoch weights | 77.4 % | 51.6 % |
| C cosine learning rate | 74.2 % | 54.8 % |
| D frozen MobileNetV2, head trained | 83.9 % | 64.5 % |
| E MobileNetV2 fine-tuned | **93.5 %** | **87.1 %** |

62 validation images means one image is 1.6 pp, so `task4_experiments.py` also
reran four of these from three different seeds — the spread reaches 11.3 pp, which
is why `REPORT_DRAFT.md` reads the three-seed means and not the table above. Its
recommendation is variant E, at 9.1 MB and ~1.7× the inference time of the model
now deployed, both measured in this page.
