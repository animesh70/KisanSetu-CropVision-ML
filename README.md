# KisanSetu CropVision ML

An independent, CPU-first visual-screening API for KisanSetu. Images are
validated and analyzed by **locally running open-source model weights**; there
is no hosted vision-inference API or paid model call. It is a separate service
so the KisanSetu backend can adopt it later without changing its current
working crop-photo workflow during this development task.

## Pipeline and safety boundary

```text
browser -> future KisanSetu backend -> CropVision /predict
image bytes -> format/size/pixel validation -> local SigLIP relevance gate
  non_crop              -> no crop or condition
  harvested_produce     -> no leaf-disease classifier
  crop_related_unclear  -> abstain
  living_crop           -> image-based crop gate -> supported leaf classifier
                           -> healthy / possible condition / abstain
```

`cropHint` is untrusted request metadata and is **not** allowed to assign a
crop or turn a non-crop image into an agricultural result. Raw bytes, filenames,
hashes, and randomness are never used for diagnosis. Public `confidence` is
always `null`: an uncalibrated softmax score is not a disease-confidence
percentage. The conservative thresholds are heuristic abstention thresholds,
not clinical/agronomic validation. Poor lighting, blur, or small subjects may
return `crop_related_unclear`.

## Models, classes, and licenses

| Role | Pinned source | License | Notes |
| --- | --- | --- | --- |
| Visual relevance + crop gate | [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224), revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Apache-2.0 | SigLIP image/text embeddings, ~0.2B parameters; CPU fp32 weights are roughly 0.8 GB before runtime overhead. Local inference only. |
| General leaf model | [Arko007/agromind-plant-disease-mobilenet](https://huggingface.co/Arko007/agromind-plant-disease-mobilenet), revision `b7c206b2426711ee9467d3991889448313f462f1` | Apache-2.0 | MobileNetV2 PlantVillage-style 38-class checkpoint. Exact ordered labels come from pinned `labels.txt`. |
| Onion training source | [Project-AgML/COLD onion leaf disease](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification), revision `72774f4bbb98b550d9f3be8c0d0e1e3d4e435fba` | CC BY 4.0 | Raw-image configuration for a future training pipeline; dataset is **not** bundled. |

The disease checkpoint supports only crops named in its 38 ordered labels.
The runtime derives `supported_crops` from those labels rather than claiming
that every KisanSetu crop is covered. Its leaf-photo training distribution may
not generalize to field photos. Image-based crop gating currently considers
Onion, Tomato, Potato, and Soybean; detecting a crop type is not the same as
supporting its disease classes. Crops outside the checkpoint labels return
`UNSUPPORTED_CROP` without a condition. **Onion has no verified pretrained or
trained custom disease checkpoint in this repository.** Harvested onions
return `HARVESTED_PRODUCE`; living onion plants return `UNSUPPORTED_CROP`.
`training/train_onion.py` implements reproducible transfer learning, but
**training pipeline implemented, checkpoint not trained**. The COLD raw set
has 815 images/four classes and only 18 raw purple-blotch images, so accuracy
claims would be premature. See [training/README.md](training/README.md).

This repository's code is MIT-licensed ([LICENSE](LICENSE)); the separately
downloaded model weights and training data retain their own licenses above.
The Python stack uses pinned PyTorch/torchvision, Transformers, Hugging Face
Hub, FastAPI, and Pillow versions; no TensorFlow stack is installed.

## API

`GET /health` is public and returns:

```json
{"status":"ok","service":"kisansetu-cropvision-ml"}
```

`POST /predict` accepts `multipart/form-data` with `image` (required) and
`cropHint` (optional). It requires `X-KisanSetu-Key`, compared to backend/Modal
secret `KISANSETU_ML_API_KEY` in constant time. Accepted **actual** image
formats are JPEG, PNG, WebP, at most 6 MB; Pillow verifies bytes and rejects
invalid/decompression-bomb/oversized images, normalizes EXIF orientation, and
converts to RGB. Example result:

```json
{
  "imageType":"non_crop",
  "crop":null,
  "assessment":"not_applicable",
  "condition":null,
  "confidence":null,
  "messageCode":"NON_CROP",
  "model":{"gate":"google/siglip-base-patch16-224","classifier":null}
}
```

`messageCode` is the localization contract for the future 12-language main
application. No price guidance is mixed into image results. Unauthorized calls
receive 401, missing image 422, invalid image 400, oversized image 413,
missing service secret/model failure 503. Internal model exceptions are not
returned to callers.

## Local setup, test, and smoke check

Use Python 3.11 or 3.12 and run from this repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:KISANSETU_ML_API_KEY = 'choose-a-local-test-key'
$env:CROPVISION_MODEL_ROOT = "$PWD\.venv\models"
.venv\Scripts\python.exe -m scripts.download_models
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check --no-cache .
.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In another terminal, use a public/safe test photo:

```powershell
.venv\Scripts\python.exe scripts\smoke_test.py --url http://127.0.0.1:8000 --key choose-a-local-test-key --image .\my-safe-photo.jpg --crop-hint Onion
```

The pinned downloads are intentionally **ignored by Git**. Model objects are
loaded once per warm process. A local request may download into the Hub cache
when `CROPVISION_MODEL_ROOT` is unset; setting it requires the downloaded
checkpoint files to be present and avoids a network call at inference time.
Tests exercise upload/security/result invariants with stubbed model decisions;
those passing tests alone do **not** establish real-world gate accuracy.

## Modal CPU deployment

The current `modal_app.py` uses Modal ASGI, a Debian Slim Python 3.11 image,
downloaded checkpoints during image build, scale-to-zero serving, **1 CPU and
2048 MiB RAM**, with no GPU. The image build needs public Hugging Face network
access. Create a Modal secret named `kisansetu-ml-api-key` containing
`KISANSETU_ML_API_KEY` before deployment. Do not put the key in Git, a browser,
or a `VITE_*` variable. Check authentication and CLI syntax, then deploy:

```powershell
.venv\Scripts\modal.exe --version
.venv\Scripts\modal.exe profile list
.venv\Scripts\modal.exe secret create kisansetu-ml-api-key KISANSETU_ML_API_KEY=YOUR_PRIVATE_KEY
.venv\Scripts\modal.exe deploy modal_app.py
```

Replace the placeholder privately; never paste a real key into chat. If no
Modal profile is authenticated, deployment is pending rather than assumed.
After actual deployment, verify the generated URL with `/health` and an
authenticated `/predict` request. The 2 GiB starting memory may need measured
adjustment because SigLIP plus Python/PyTorch memory can exceed weight size.
Modal scale-to-zero and model download make cold starts slower than warm calls.

Future KisanSetu backend variables are `CROP_VISION_URL` and
`CROP_VISION_API_KEY`; the browser must call the **KisanSetu backend**, which
then calls this service. The ML key must remain server-side. This repository
does not edit or deploy the main KisanSetu application.

## Limitations and disclaimer

Zero-shot relevance prompting is not a calibrated agricultural classifier.
Photographs of harvested produce, living crops, documents, artwork, and blurry
scenes need a separately measured regression set; abstention is preferable to
inventing a disease. The PlantVillage-style checkpoint is not a field-tested
diagnostic. Onion disease classification is unavailable until a trustworthy
checkpoint is trained and validated. This performs **preliminary visual
screening, not a definitive agricultural diagnosis**. Confirm disease and
treatment with a qualified agriculture expert before acting.
