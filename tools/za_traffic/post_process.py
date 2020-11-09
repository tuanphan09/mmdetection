import os
import json
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Ready for submition')

parser.add_argument('--json_in', dest='json_in', type=str, required=True,
                    help="File result")


parser.add_argument('--json_out', dest='json_out', type=str, required=True,
                    help="File result")

args = parser.parse_args()

def save_coco(file, info, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def main(args):
    with open(args.json_in, 'rt', encoding='UTF-8') as json_file:
        annotations = json.load(json_file)
        for i, ann in enumerate(annotations):
            # print(ann)
            ann["segmentation"] = []
            ann["area"] = 0
            ann["iscrowd"] = 0
            ann["id"] = i

    with open(args.json_out, 'wt', encoding='UTF-8') as outfile:
        json.dump(annotations, outfile)

main(args)