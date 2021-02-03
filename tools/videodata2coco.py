# -*- coding: utf-8 -*-
import mmcv
from tqdm import tqdm


ind2cls = {
    1: 'BaseballPitch',
    2: 'BasketballDunk',
    3: 'Billiards',
    4: 'CleanAndJerk',
    5: 'CliffDiving',
    6: 'CricketBowling',
    7: 'CricketShot',
    8: 'Diving',
    9: 'FrisbeeCatch',
    10: 'GolfSwing',
    11: 'HammerThrow',
    12: 'HighJump',
    13: 'JavelinThrow',
    14: 'LongJump',
    15: 'PoleVault',
    16: 'Shotput',
    17: 'SoccerPenalty',
    18: 'TennisSwing',
    19: 'ThrowDiscus',
    20: 'VolleyballSpiking'}


cls2ind = {
    'BaseballPitch': 1,
    'BasketballDunk': 2,
    'Billiards': 3,
    'CleanAndJerk': 4,
    'CliffDiving': 5,
    'CricketBowling': 6,
    'CricketShot': 7,
    'Diving': 8,
    'FrisbeeCatch': 9,
    'GolfSwing': 10,
    'HammerThrow': 11,
    'HighJump': 12,
    'JavelinThrow': 13,
    'LongJump': 14,
    'PoleVault': 15,
    'Shotput': 16,
    'SoccerPenalty': 17,
    'TennisSwing': 18,
    'ThrowDiscus': 19,
    'VolleyballSpiking': 20}


FPS = 25


def construct_imginfo(filename, h, w, ID):
    image = {"license": 1,
             "file_name": filename,
             "coco_url": "xxx",
             "height": h,
             "width": w,
             "date_captured": "2019-06-25",
             "flickr_url": "xxx",
             "id": ID
             }
    return image


def construct_ann(obj_id, ID, category_id, seg, area, bbox):
    ann = {"id": obj_id,
           "image_id": ID,
           "category_id": category_id,
           "segmentation": seg,
           "area": area,
           "bbox": bbox,
           "iscrowd": 0,
           }
    return ann


def generate_coco(annos, out_file):
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind + 1, "name": ind2cls[ind + 1], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    img_names = {}

    IMG_ID = 0
    OBJ_ID = 0
    for key, info in tqdm(annos.items()):
        h, w = 1, int(info['duration_second'] * FPS)
        if key not in img_names:
            img_names[key] = IMG_ID
            img_info = construct_imginfo("defect", h, w, IMG_ID)
            annotations["images"].append(img_info)
            IMG_ID = IMG_ID + 1

        for ann in info['annotations']:
            img_id = img_names[key]
            xmin, xmax = ann['segment']
            xmin = int(xmin * FPS)
            xmax = int(xmax * FPS)
            ymin, ymax = 0, 1
            label = ann['label']
            cat_ID = cls2ind[label]

            area = (ymax - ymin) * (xmax - xmin)
            seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            ann = construct_ann(OBJ_ID, img_id, cat_ID, seg, area, bbox)
            annotations["annotations"].append(ann)
            OBJ_ID += 1

    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)


if __name__ == "__main__":
    anns = mmcv.load("/DATA/data/public/TAD/thumos14/annotations_thumos14_20cls_test.json")
    print("convert to coco format...")
    generate_coco(anns, 'data/thumos14_test_coco_fps25.json')
