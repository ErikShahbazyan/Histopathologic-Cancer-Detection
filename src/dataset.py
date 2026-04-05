import pandas as pd
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from src import CFG

# ── Transforms ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Datasets ───────────────────────────────────────────
class CancerDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.img_dir / f"{row['id']}.tif").convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_ids   = [f.stem for f in Path(img_dir).glob("*.tif")]
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img    = Image.open(self.img_dir / f"{img_id}.tif").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id


# ── DataLoader builders ────────────────────────────────
def get_train_val_loaders():
    df = pd.read_csv(CFG["paths"]["labels_csv"])
    train_df, val_df = train_test_split(
        df,
        test_size=CFG["data"]["val_size"],
        stratify=df["label"],
        random_state=CFG["training"]["seed"],
    )
    train_dataset = CancerDataset(train_df, CFG["paths"]["train_dir"], transform=train_transform)
    val_dataset   = CancerDataset(val_df,   CFG["paths"]["train_dir"], transform=val_transform)

    bs = CFG["training"]["batch_size"]
    nw = CFG["data"]["num_workers"]

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader, df


def get_test_loader():
    test_dataset = TestDataset(CFG["paths"]["test_dir"], transform=val_transform)
    return DataLoader(
        test_dataset,
        batch_size=CFG["training"]["batch_size"],
        shuffle=False,
        num_workers=CFG["data"]["num_workers"],
    )
