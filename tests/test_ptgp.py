import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from ptgp import PTGP, make_balanced_groups
from ptgp.data import load_dataset_splits


class PTGPTests(unittest.TestCase):
    def test_projection_is_orthogonal_and_does_not_amplify(self):
        images = torch.randn(12, 3, 4, 4)
        labels = torch.tensor([0, 1] * 6)
        dataset = TensorDataset(images, labels)
        groups = make_balanced_groups(dataset, per_class=4, groups=4, seed=9)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
        projector = PTGP(dataset, groups, microbatch_size=2)
        projector.fit(model)
        gradients = torch.autograd.grad(
            torch.nn.functional.cross_entropy(model(images[:2]), labels[:2]),
            tuple(model.parameters()))
        projected, diagnostic = projector.project(gradients)
        raw = torch.cat([item.flatten() for item in gradients])
        defended = torch.cat([item.flatten() for item in projected])
        self.assertLessEqual(diagnostic["retained_energy_fraction"], 1.000001)
        self.assertTrue(torch.allclose(torch.dot(raw, defended),
                                       torch.dot(defended, defended), atol=1e-5))

    def test_imagenet_reference_is_train_and_evaluation_is_val(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "train").mkdir()
            (base / "val").mkdir()
            class FakeFolder:
                def __init__(self, path, transform=None):
                    self.path = Path(path)
                    self.classes = [f"class{i}" for i in range(1000)]
                    self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            with patch("ptgp.data.datasets.ImageFolder", FakeFolder):
                training, reference, evaluation = load_dataset_splits("imagenet", directory)
            self.assertIs(training, reference)
            self.assertEqual(training.path.name, "train")
            self.assertEqual(evaluation.path.name, "val")

    def test_imagenet_never_uses_val_as_training_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "val").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "val/ is never used"):
                load_dataset_splits("imagenet", directory)


if __name__ == "__main__":
    unittest.main()
