import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from clip_reproduction.datasets import get_dataset
from clip_reproduction.models.vision import CNNModel

epochs = 1
lr = 0.1
batch_size = 10
evaluate_every = 1
val_ratio = 0.1  # 10% validation
num_workers = 4


generator = torch.Generator().manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = get_dataset("cifar100", train=True)

n_total = len(dataset)
n_val = int(val_ratio * n_total)
n_train = n_total - n_val
nb_batches = len(dataset) // batch_size

train_dataset, val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=generator,
    num_workers=num_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss = 0
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss += criterion(outputs, labels)

    model.train()
    return loss / len(loader)


model = CNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for e in range(epochs):
    print(f"\n========== Epoch {e + 1:03d}/{epochs:03d} ==========")

    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % max(1, nb_batches // 5) == 0:
            avg_loss = running_loss / (i + 1)
            print(f"[Train] Batch {i:04d}/{nb_batches:04d} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

    epoch_train_loss = running_loss / nb_batches
    print(f"[Train] Epoch loss: {epoch_train_loss:.4f}")

    if (e + 1) % evaluate_every == 0:
        print("\n----------------- Evaluation -----------------")
        model.eval()
        val_loss = evaluate(model, val_loader, criterion)
        print(f"[Val]   Loss: {val_loss:.4f}")
        print("----------------------------------------------")
