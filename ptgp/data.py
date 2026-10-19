"""Explicit dataset splits for training, task probes, and evaluation."""

from pathlib import Path

from torchvision import datasets, transforms


def _image_transform(size: int, mean, std):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_dataset_splits(name: str, root: str, *, image_size: int = 64):
    """Return (training, PTGP reference, evaluation) datasets.

    ImageNet requires actual train/ and val/ directories. In particular, val/
    is never silently substituted for missing training images. The reference
    dataset is val/, as specified by this release's protocol. Callers must
    exclude target indices from that reference set for a privacy evaluation.
    """
    root = Path(root).expanduser().resolve()
    name = name.lower().replace("-", "")
    if name in {"cifar10", "cifar100"}:
        dataset_class = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        mean, std = (
            ((.4914672375, .4822617471, .4467701316),
             (.2470322400, .2434851378, .2615878582))
            if name == "cifar10" else
            ((.5071598291, .4866936207, .4412019253),
             (.2673342824, .2564384639, .2761504650))
        )
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        train = dataset_class(str(root), train=True, download=False, transform=transform)
        test = dataset_class(str(root), train=False, download=False, transform=transform)
        # CIFAR experiments use a class-balanced training reference, matching
        # the server's final PTGP implementation; evaluation uses the test set.
        return train, train, test
    if name == "imagenet":
        train_root, val_root = root / "train", root / "val"
        if not train_root.is_dir() or not val_root.is_dir():
            raise FileNotFoundError(
                f"ImageNet needs both {train_root} and {val_root}; "
                "val/ is never used as a training fallback."
            )
        transform = _image_transform(image_size, (.485, .456, .406), (.229, .224, .225))
        train = datasets.ImageFolder(str(train_root), transform=transform)
        val = datasets.ImageFolder(str(val_root), transform=transform)
        if len(train.classes) != 1000 or train.class_to_idx != val.class_to_idx:
            raise ValueError("ImageNet train/ and val/ must share the same 1,000 WNID classes")
        return train, val, val
    if name == "ffhq":
        train_root, eval_root = root / "train", root / "eval"
        if not train_root.is_dir() or not eval_root.is_dir():
            raise FileNotFoundError("FFHQ needs labeled train/ and eval/ directories")
        transform = _image_transform(image_size, (.5, .5, .5), (.5, .5, .5))
        train = datasets.ImageFolder(str(train_root), transform=transform)
        test = datasets.ImageFolder(str(eval_root), transform=transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("FFHQ train/ and eval/ age-label class mapping differs")
        return train, train, test
    raise ValueError(f"Unknown dataset: {name}")
