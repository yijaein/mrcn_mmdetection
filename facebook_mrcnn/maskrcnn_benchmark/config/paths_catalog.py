# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/home/bong07/data"
    DATASETS = {
        # `COCO_2017_train` = `COCO_2014_train` + `valminusminival` , `COCO_2017_val` = `minival`
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "rsna_512_train": {
            "img_dir": "rsna/rsna512/train",
            "ann_file": "rsna/rsna512/train_labels_512.csv"
        },
        "rsna_512_test": {
            "img_dir": "rsna/rsna512/test",
            "ann_file": "rsna/rsna512/test_labels_512.csv"
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        "kidney_isangmi_400+100+1200_withExcluded_train": {
            "img_dir": "yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_20190301"
        },
        "kidney_isangmi_400+100+1200_withExcluded_val": {
            "img_dir": "yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_20190301"
        },
        "kidney_isangmi_400+100+1200_ckd_normal_train": {
            "img_dir": "yonsei2/dataset/US_isangmi_400+100+1200_for_classification/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_20190227_3classes"
        },
        "kidney_isangmi_400+100+1200_ckd_normal_val": {
            "img_dir": "yonsei2/dataset/US_isangmi_400+100+1200_for_classification/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_20190227_3classes"
        },
        "kidney_isangmi_400+100+1200_aki_ckd_train": {
            "img_dir": "yonsei2/dataset/US_isangmi_400+100+1200_for_classification/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_20190227_3classes"
        },
        "kidney_isangmi_400+100+1200_aki_ckd_val": {
            "img_dir": "yonsei2/dataset/US_isangmi_400+100+1200_for_classification/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_20190227_3classes"
        },
        "kidney_liver_isangmi_some_400_withExcluded_train": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190315/us/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190315/seg"
        },
        "kidney_liver_isangmi_some_400_withExcluded_val": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190315/us/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190315/seg"
        },

        "liver_isangmi_some_400_withExcluded_train": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/us/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/seg"
        },
        "liver_isangmi_some_400_withExcluded_val": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/us/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/seg"
        },

        "liver_bg_isangmi_some_400_withExcluded_train": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/us/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/seg"
        },
        "liver_bg_isangmi_some_400_withExcluded_val": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/us/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/seg"
        },

        "liver_bg_augmask_isangmi_some_400_withExcluded_train": {
            "img_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404 (copy)/us/train",
            "mask_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404 (copy)/seg"
        },
        "liver_bg_augmask_isangmi_some_400_withExcluded_val": {
            "img_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404 (copy)/us/val",
            "mask_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404 (copy)/seg"
        },

        "liver_bg_augpolygon_isangmi_some_400_withExcluded_train": {
            "img_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/us/train",
            "mask_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/seg"
        },
        "liver_bg_augpolygon_isangmi_some_400_withExcluded_val": {
            "img_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/us/val",
            "mask_dir": "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegMrcnn_kidney_liver_bg_some_400_20190404/seg"
        },

        "liver_only_only_train": {
            "img_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527 (liveronly)/us/train",
            "mask_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527 (liveronly)/seg"
        },
        "liver_only_only_val": {
            "img_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527 (liveronly)/us/val",
            "mask_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527 (liveronly)/seg"
        },

        "liver_400_100_train": {
            "img_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527/us/train",
            "mask_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527/seg"
        },
        "liver_400_100_val": {
            "img_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527/us/val",
            "mask_dir": "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527/seg"
        },

        "liver_isangmi_test_train": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/us/train",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/seg"
        },
        "liver_isangmi_test_val": {
            "img_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/us/val",
            "mask_dir": "yonsei2/dataset/SegMrcnn_kidney_liver_some_400_20190403/seg"
        },
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "rsna" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                # mask_type= "image"
                mask_type="polygon"
            )
            return dict(
                factory="RSNADataset",
                args=args,
            )
        elif "kidney_isangmi_400+100+1200_withExcluded_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="KidneyDataset",
                args=args,
            )
        elif "kidney_isangmi_400+100+1200_ckd_normal_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="KidneyCkdNorDataset",
                args=args,
            )
        elif "kidney_isangmi_400+100+1200_aki_ckd_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="KidneyAkiCkdDataset",
                args=args,
            )
        elif "kidney_liver_isangmi_some_400_withExcluded_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="KidneyLiverDataset",
                args=args,
            )
        elif "liver_isangmi_some_400_withExcluded_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="LiverDataset",
                args=args,
            )
        elif "liver_bg_isangmi_some_400_withExcluded_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="LiverBgDataset",
                args=args,
            )
        elif "liver_bg_augmask_isangmi_some_400_withExcluded_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="LiverBgAugmaskDataset",
                args=args,
            )
        elif "liver_bg_augpolygon_isangmi_some_400_withExcluded_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            return dict(
                factory="LiverBgAugpolygonDataset",
                args=args,
            )
        elif "liver_isangmi_test_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            print('*factor liver_isangmi_test_')
            return dict(
                factory="LiverTestDataset",
                args=args,
            )

        elif "liver_400_100_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            print('*factor liver_isangmi_test_')
            return dict(
                factory="LiverCombineAugpolygonDataset",
                args=args,
            )

        elif "liver_only_only_" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                mask_dir=os.path.join(data_dir, attrs["mask_dir"]),
                mask_type="image"
            )
            print('*factor liver_isangmi_test_')
            return dict(
                factory="LiverOnlyAugpolygonDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
