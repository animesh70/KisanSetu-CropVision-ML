"""Reproducible transfer learning on raw COLD onion leaves; no made-up metrics."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 26132
DATASET_ID = "Project-AgML/COLD_onion_leaf_disease_classification"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class OnionDataset(Dataset):
    def __init__(self, rows, transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return self.transform(row["image"].convert("RGB")), int(row["label"])


def split_raw_data(rows):
    # The published "augmented" configuration is intentionally never used.
    first_split = rows.train_test_split(
        test_size=0.20, seed=SEED, stratify_by_column="label"
    )
    train, held = first_split["train"], first_split["test"]
    second_split = held.train_test_split(
        test_size=0.50, seed=SEED, stratify_by_column="label"
    )
    validation, test = second_split["train"], second_split["test"]
    return train, validation, test


def transforms_for_training():
    common = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            *common,
        ]
    )
    evaluation = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), *common]
    )
    return train, evaluation


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    predicted: list[int] = []
    actual: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(loss_fn(logits, labels)) * len(labels)
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.cpu().tolist())
    return total_loss / len(actual), actual, predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/onion"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rows = load_dataset(DATASET_ID, "raw", split="train")
    labels = rows.features["label"].names
    train_rows, val_rows, test_rows = split_raw_data(rows)
    train_aug, eval_transform = transforms_for_training()
    train_loader = DataLoader(
        OnionDataset(train_rows, train_aug), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        OnionDataset(val_rows, eval_transform), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        OnionDataset(test_rows, eval_transform), batch_size=args.batch_size
    )

    counts = Counter(int(label) for label in train_rows["label"])
    class_weights = torch.tensor(
        [len(train_rows) / (len(labels) * counts[index]) for index in range(len(labels))],
        dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best_onion_state.pth"
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(targets)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_rows),
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, actual, predicted = evaluate(model, test_loader, loss_fn, device)
    report = classification_report(
        actual, predicted, labels=list(range(len(labels))), target_names=labels, output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist()
    results = {
        "dataset": DATASET_ID,
        "configuration": "raw",
        "seed": SEED,
        "classes": labels,
        "split_sizes": {
            "train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)
        },
        "history": history,
        "test_loss": test_loss,
        "classification_report": report,
        "confusion_matrix": matrix,
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"test_loss": test_loss, "report": report, "matrix": matrix}, indent=2))


if __name__ == "__main__":
    main()
