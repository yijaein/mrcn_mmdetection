import json

from collections import OrderedDict


target_own = "/home/bong3/data/iitp/track1/iitp_annotation_merge_vis_1920.json"
target_two = "/home/bong3/data/iitp/track1/coco_annotation_no_fire.json"
# target_two = "/home/bong3/data/iitp/track1/vis_annotation.json"

out_path = "/home/bong3/data/iitp/track1/iitp_annotation_merge_vis_coco_1920.json"

f1 = open(target_own, 'r', encoding='utf-8')
ann_info = json.load(f1, object_pairs_hook=OrderedDict)

# ann_info = ann_info*4

f2 = open(target_two, 'r', encoding='utf-8')
merge_ann = json.load(f2, object_pairs_hook=OrderedDict)

f1.close()
f2.close()

ann_info.extend(merge_ann)

with open(out_path, 'w', encoding='utf-8') as json_file:
    json.dump(ann_info, json_file, ensure_ascii=False, indent='\t')


# test = "/home/bong3/data/iitp/track1/iitp_annotation_merge.json"
# f3 = open(test, 'r', encoding='utf-8')
# test_info = json.load(f3, object_pairs_hook=OrderedDict)
# f3.close()
# print(len(ann_info), len(merge_ann), len(test_info))