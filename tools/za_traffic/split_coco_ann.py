import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('--train_size', dest='split', type=float, required=True,
                    help="A percentage of a training set; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

args = parser.parse_args()

def save_coco(file, info, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
            print("Remove images without annotations: {}/{} removed".format(number_of_images - len(images), number_of_images))

        x, y = train_test_split(images, train_size=args.split)

        save_coco(args.train, info, x, filter_annotations(annotations, x), categories)
        save_coco(args.test, info, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))

# python3 cocosplit.py --having-annotations --train_size 0.8 data/train.json tmp/train.json tmp/test.json
main(args)