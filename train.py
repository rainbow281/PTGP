"""Compact FedAvg training entry point with a PTGP gradient transform."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ptgp import PTGP, load_dataset_splits, make_balanced_groups
from ptgp.model import make_model


def partition(labels, clients, alpha, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    result = [[] for _ in range(clients)]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
        for client, split in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
            result[client].extend(split.tolist())
    return result


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    loss = correct = count = 0
    for images, labels in DataLoader(dataset, batch_size=128, shuffle=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    return loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "ffhq"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["ptgp", "none"], default="ptgp")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.1)
    parser.add_argument("--dirichlet-alpha", type=float, default=.5)
    parser.add_argument("--reference-per-class", type=int)
    parser.add_argument("--probe-groups", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training, reference, evaluation = load_dataset_splits(args.dataset, str(args.data_root))
    labels = getattr(training, "targets", None)
    if labels is None:
        labels = [label for _, label in training.samples]
    shards = partition(labels, args.clients, args.dirichlet_alpha, args.seed)
    if min(map(len, shards)) == 0:
        raise ValueError("At least one client has no training examples; adjust the partition")
    classes = len(set(map(int, labels)))
    per_class = args.reference_per_class or 200
    groups = make_balanced_groups(reference, per_class=per_class,
                                  groups=args.probe_groups, seed=args.seed)
    train_mode_probe = args.dataset in {"cifar10", "ffhq"}
    projector = PTGP(reference, groups,
                     microbatch_size=max(map(len, groups)) if train_mode_probe else 32,
                     probe_mode="train" if train_mode_probe else "eval")
    model = make_model(classes, args.seed).to(device)
    metadata = vars(args).copy()
    metadata.update(data_root=str(args.data_root), output=str(args.output),
                    training_split="train", reference_split="train",
                    evaluation_split="val" if args.dataset == "imagenet" else "test/eval",
                    model="random CIFAR-style ResNet-18", reference_images=per_class * classes)
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    with (args.output / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["round", "train_loss", "train_accuracy",
                                                    "evaluation_loss", "evaluation_accuracy"])
        writer.writeheader()
        for round_index in range(1, args.rounds + 1):
            if args.method == "ptgp":
                projector.fit(model)
            initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
            accumulated = {key: torch.zeros_like(value) for key, value in initial.items()
                           if value.is_floating_point()}
            selected = np.random.default_rng(args.seed + round_index).choice(
                args.clients, args.clients_per_round, replace=False)
            total_loss = correct = count = 0
            for client in selected:
                model.load_state_dict(initial)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
                loader = DataLoader(Subset(training, shards[int(client)]),
                                    batch_size=args.batch_size, shuffle=True)
                for images, labels in loader:
                    model.train()
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                    raw = torch.autograd.grad(loss, tuple(model.parameters()))
                    defended = projector.project(raw)[0] if args.method == "ptgp" else raw
                    optimizer.zero_grad(set_to_none=True)
                    for parameter, gradient in zip(model.parameters(), defended):
                        parameter.grad = gradient
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(labels)
                    correct += int((logits.argmax(1) == labels).sum())
                    count += len(labels)
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        if key in accumulated:
                            accumulated[key].add_(value / len(selected))
            model.load_state_dict({key: accumulated.get(key, value)
                                   for key, value in initial.items()})
            val_loss, val_accuracy = evaluate(model, evaluation, device)
            row = dict(round=round_index, train_loss=total_loss / count,
                       train_accuracy=correct / count, evaluation_loss=val_loss,
                       evaluation_accuracy=val_accuracy)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
