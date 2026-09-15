# Onion leaf classifier training

The API deliberately **does not classify onion diseases yet**. No onion checkpoint
has been trained or verified for this repository.

The training script uses the **raw**, not augmented, configuration of
[Project-AgML/COLD_onion_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification)
(CC BY 4.0). It has 815 raw onion-leaf images, four classes, and only 18 raw
purple-blotch samples. This severe imbalance makes class-level reliability
uncertain even after transfer learning.

Run from the repository root after installing `requirements-dev.txt`:

```powershell
python training/train_onion.py --epochs 20 --output checkpoints/onion
```

The script splits **raw images before train-only augmentation**, fixes a random
seed, uses stratified train/validation/test partitions, balances the loss,
early-stops on validation loss, saves the best checkpoint, and writes a
classification report plus confusion matrix from the untouched test split.
Those reports are ignored by Git. They are not generated or claimed unless
training actually runs.

Before integrating any checkpoint, independently review duplicate-source
leakage, per-class precision/recall (especially purple blotch), calibration,
out-of-distribution field photos, and horticultural expert validation.
