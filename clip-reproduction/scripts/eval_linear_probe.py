from pathlib import Path

import hydra
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clip_reproduction import utils
from clip_reproduction.datasets import get_classification_train_test_datasets
from clip_reproduction.models.factory import create_model

MODEL_NAME = "resnet50"


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


@torch.no_grad()
def compute_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        features = model.encode_image_penultimate(images)
        all_features.append(features.cpu().numpy().astype(np.float32))
        all_labels.append(labels.numpy())

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return features_np, labels_np


def cache_path(cache_dir: Path, dataset_name: str, split: str) -> Path:
    return cache_dir / f"{dataset_name}_{MODEL_NAME}_{split}_features.npz"


def load_or_compute_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cache_dir: Path,
    dataset_name: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(cache_dir=cache_dir, dataset_name=dataset_name, split=split)
    if path.exists():
        data = np.load(path)
        return data["features"], data["labels"]

    features, labels = compute_features(model=model, loader=loader, device=device)
    np.savez_compressed(path, features=features, labels=labels)
    return features, labels


@hydra.main(version_base=None, config_path="../conf", config_name="eval_linear_probe_conf")
def main(cfg) -> None:
    utils.seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    print(f"Using device: {device}")

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, num_classes, _ = get_classification_train_test_datasets(
        name=cfg.dataset,
        root=cfg.data_root,
        image_size=cfg.image_size,
        download=True,
    )

    model = create_model(
        name=MODEL_NAME,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    x_train, y_train = load_or_compute_features(
        model=model,
        loader=train_loader,
        device=device,
        cache_dir=cache_dir,
        dataset_name=cfg.dataset,
        split="train",
    )
    x_test, y_test = load_or_compute_features(
        model=model,
        loader=test_loader,
        device=device,
        cache_dir=cache_dir,
        dataset_name=cfg.dataset,
        split="test",
    )

    x_train = l2_normalize(x_train)
    x_test = l2_normalize(x_test)

    # sklearn uses C = 1/lambda for L2 strength.
    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=cfg.logreg_c,
        max_iter=1000,
        n_jobs=None,
        random_state=cfg.seed,
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
