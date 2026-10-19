#!/usr/bin/env python3
"""Validate an ImageNet-1K ImageFolder tree before formal inversion experiments."""

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {'.jpeg', '.jpg', '.png'}


def image_files(root):
    return sorted(
        path for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-train', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--class-index-json', default=None)
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    val_root = data_path / 'val'
    train_root = data_path / 'train'
    if not val_root.is_dir():
        raise SystemExit(f'Missing validation directory: {val_root}')
    if args.require_train and not train_root.is_dir():
        raise SystemExit(f'Missing training directory: {train_root}')

    val_classes = sorted(path.name for path in val_root.iterdir() if path.is_dir())
    val_images = image_files(val_root)
    if len(val_classes) != 1000:
        raise SystemExit(
            f'Expected 1000 ImageNet-1K validation classes, found {len(val_classes)}')
    if args.strict and len(val_images) != 50000:
        raise SystemExit(
            f'Expected exactly 50000 ImageNet-1K validation images, found {len(val_images)}')

    train_classes = []
    train_count = None
    if train_root.is_dir():
        train_classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
        if train_classes != val_classes:
            raise SystemExit('train/ and val/ have different synset directory mappings')
        if args.strict or args.require_train:
            train_count = len(image_files(train_root))
            if args.strict and train_count != 1281167:
                raise SystemExit(
                    f'Expected exactly 1281167 ImageNet-1K training images, found {train_count}')

    if min(args.indices) < 0 or max(args.indices) >= len(val_images):
        raise SystemExit(
            f'Indices must be inside [0, {len(val_images) - 1}], got {args.indices}')

    official_mapping_checked = False
    if args.class_index_json:
        mapping_path = Path(args.class_index_json).expanduser().resolve()
        with mapping_path.open('r', encoding='utf-8') as stream:
            raw_mapping = json.load(stream)
        official_wnids = [raw_mapping[str(index)][0] for index in range(1000)]
        numeric_classes = val_classes == [f'{index:04d}' for index in range(1000)]
        if not numeric_classes and official_wnids != val_classes:
            mismatches = [
                (index, official_wnids[index], val_classes[index])
                for index in range(1000)
                if official_wnids[index] != val_classes[index]
            ][:10]
            raise SystemExit(
                'ImageFolder ordering does not match the official ImageNet-1K/BigGAN '
                f'class order. First mismatches: {mismatches}')
        official_mapping_checked = True

    selected = []
    for index in args.indices:
        path = val_images[index]
        selected.append({
            'index': index,
            'relative_path': str(path.relative_to(data_path)),
            'synset': path.parent.name,
            'imagefolder_label': val_classes.index(path.parent.name),
        })

    manifest = {
        'dataset': 'ILSVRC2012 ImageNet-1K',
        'data_path': str(data_path),
        'validation_classes': len(val_classes),
        'validation_images': len(val_images),
        'training_classes': len(train_classes) if train_classes else None,
        'training_images': train_count,
        'strict_validation': bool(args.strict),
        'official_class_mapping_checked': official_mapping_checked,
        'class_order_sha256': hashlib.sha256(
            '\n'.join(val_classes).encode('utf-8')).hexdigest(),
        'selected_samples': selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
