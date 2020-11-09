import os
import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Create COCO data from list images')
parser.add_argument('--example', dest='example', type=str, required=True,
                    help="Annotation example")

parser.add_argument('--img_dir', dest='img_dir', type=str, required=True,
                    help="Images directory")


parser.add_argument('--json_out', dest='json_out', type=str, required=True,
                    help="Where to save file json")

args = parser.parse_args()

def save_coco(file, info, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def main(args):
    with open(args.example, 'rt', encoding='UTF-8') as example:
        coco = json.load(example)
        info = coco['info']
        categories = coco['categories']

        images = []
        annotations = []
        for i, fname in enumerate(os.listdir(args.img_dir)):
            img_id, ext = fname.split('.')
            img_id = int(img_id)
            if ext == "png":
                img_info = {
                    "file_name": fname,
                    "height": 626,
                    "id": img_id,
                    "street_id": 0,
                    "width": 1622
                    }
                images.append(img_info)

                ann = {
                    "area": 0,
                    "bbox": [0, 0, 40, 40],
                    "category_id": 1,
                    "id": i,
                    "image_id": img_id,
                    "iscrowd": 0,
                    "segmentation": []
                }
                annotations.append(ann)
        save_coco(args.json_out, info, images, annotations, categories)

main(args)