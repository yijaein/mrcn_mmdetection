import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from mmdetection.tools.convert_datasets import RSNAConvert, VisDroneConvert


# python3 make_annotation.py '/home/bong3/data/rsna512/train' -ap '/home/bong3/data/rsna512/train_labels_512.csv' -o '/home/bong3/data/rsna512/train_ann.json' -type 'rsna'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ours dataset annotations to mmdetection format')
    parser.add_argument('dataset_root_path', type=str, help='dataset root path')
    parser.add_argument('-ap', '--ann-path', type=str, default=None, help='Convert annotation to mmd format')
    parser.add_argument('-mp', '--mask-root-path', type=str, default=None, help='Convert mask to annotation')
    parser.add_argument('-o', '--out-path', type=str, help='output path')
    parser.add_argument('-type', '--dataset-type', type=str, help='Select dataset type.[rsna, kiha, vis]')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_type = (args.dataset_type).lower()

    out_path = args.out_path
    if os.path.splitext(out_path)[-1] == '':
        raise ValueError("Need filename and '.json' from output path")

    if dataset_type == 'rsna':
        convert = RSNAConvert(args.dataset_root_path, args.out_path, ann_path=args.ann_path)
    elif dataset_type == 'kiha':
        raise ValueError("Not created kiha converter...")
    elif dataset_type == 'vis':
        convert = VisDroneConvert(args.dataset_root_path, args.out_path, ann_path=args.ann_path)

    convert.build()
    convert.save_ann_file()
    print('Done!')


if __name__ == '__main__':
    main()