import json 
from tqdm import tqdm
from pathlib import Path 

WANTED_CLASS = ['person']

# BCNET_COCO_JSON = '/data/PublicDatasets/COCO2017/annotations/bcnet/instances_train_2017_transform_slight_correct.json'
# OUT_APPEND = "-person"
# KEEP_NEGATIVE = False

BCNET_COCO_JSON = '/data/PublicDatasets/COCO2017/annotations/instances_train2017.json'
OUT_APPEND = "-person"
KEEP_NEGATIVE = False

# BCNET_COCO_JSON = '/data/PublicDatasets/COCO2017/annotations/instances_val2017.json'
# OUT_APPEND = "-5k-person"
# KEEP_NEGATIVE = True

with open(BCNET_COCO_JSON, 'r') as rf: 
    coco_dict = json.load(rf)

cats = []
wanted_cat_ids = []
for cat_dict in coco_dict['categories']: 
    if cat_dict['name'] in WANTED_CLASS: 
        wanted_cat_ids.append(cat_dict['id'])
        cats.append(cat_dict)

new_annots = []
wanted_img_ids = []
for annot_dict in tqdm(coco_dict['annotations']):
    if annot_dict['category_id'] in wanted_cat_ids: 
        if annot_dict['image_id'] not in wanted_img_ids:
            wanted_img_ids.append(annot_dict['image_id'])
        new_annots.append(annot_dict)

if KEEP_NEGATIVE:
    new_imgs = coco_dict['images']
else:
    new_imgs = []
    for img_dict in tqdm(coco_dict['images']):
        if img_dict['id'] in wanted_img_ids: 
            img_dict['file_name'] = Path(img_dict['file_name']).name
            new_imgs.append(img_dict)

print(f"num categories: {len(coco_dict['categories'])} -> {len(cats)}")
print(f"num annotations: {len(coco_dict['annotations'])} -> {len(new_annots)}")
print(f"num images: {len(coco_dict['images'])} -> {len(new_imgs)}")

coco_dict['categories'] = cats
coco_dict['annotations'] = new_annots
coco_dict['images'] = new_imgs


out_json = Path(BCNET_COCO_JSON)
out_json = out_json.parent / f"{out_json.stem}{OUT_APPEND}.json"

with out_json.open('w') as wf: 
    json.dump(coco_dict, wf)

print(f'filtered json dumped to {out_json}')