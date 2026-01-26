import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split

epochs = 1
lr = 0.1
batch_size = 10
evaluate_every = 1
val_ratio = 0.1  # 10% validation


generator = torch.Generator().manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = torchvision.datasets.CIFAR100(
    root="./data/raw", transform=torchvision.transforms.ToTensor(), download=True
)

n_total = len(dataset)
n_val = int(val_ratio * n_total)
n_train = n_total - n_val
nb_batches = len(dataset) // batch_size


dataloader = DataLoader(dataset, batch_size=batch_size)
train_dataset, val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=generator,
)
train_loader, val_loader = (
    DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    ),
)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "CNN"
        # -------- Feature extractor --------
        self.features = nn.Sequential(
            # Block 1: 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            # Block 2: 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            # Block 3: 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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
