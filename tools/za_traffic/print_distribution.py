import json
import argparse

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')

args = parser.parse_args()


def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        data = {}
        for cat in categories:
            data[cat['id']] = {'name': cat['name'], 'count': 0}

        for ann in annotations:
            data[ann['category_id']]['count'] += 1
        
        for c in data:
            print("{}\t{}".format(data[c]['count'], data[c]['name']))

main(args)