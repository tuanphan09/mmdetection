import os
import json
import argparse
from sklearn.model_selection import train_test_split

from ensemble_boxes import *

parser = argparse.ArgumentParser(description='Ready for submition')

parser.add_argument('--json_in', dest='json_in', type=str, required=True,
                    help="File result")

parser.add_argument('--weights', dest='weights', type=str, required=True,
                    help="File result")

parser.add_argument('--json_out', dest='json_out', type=str, required=True,
                    help="File result")

args = parser.parse_args()

iou_thr = 0.5
skip_box_thr = 0.01
sigma = 0.1

height = 622
width = 1622


def main(args):
    list_json_in = args.json_in.split(',')
    
    weights = [float(w) for w in args.weights.split(',')]

    if len(list_json_in) != len(weights):
        print("list jsons and list weights must be equal in length")
        exit()

    
    list_images_id = []
    list_anns = []
    for json_in in list_json_in:
        with open(json_in, 'rt', encoding='UTF-8') as json_file:
            annotations = json.load(json_file)
            new_annotations = []
            for ann in annotations:
                imgage_id = ann['image_id']
                x, y, w, h = ann['bbox']
                if x + w > width or y + h > height or w * h <= 4:
                    print("skip", ann)
                    continue
                ann['bbox'] = [x / width, y / height, (x + w) / width, (y + h) / height] # normalize [0, 1] - [x1, y1, x2, y2]
                new_annotations.append(ann)
                if imgage_id not in list_images_id:
                    list_images_id.append(imgage_id)
            
            list_anns.append(new_annotations)

    print("Number of images: {}".format(len(list_images_id)))
    final_result = []
    for imgage_id in list_images_id:
        boxes_list = []
        scores_list = []
        labels_list = []
        for annotations in list_anns:
            bboxes = []
            scores = []
            labels = []
            for ann in annotations:
                if ann['image_id'] == imgage_id:
                    bboxes.append(ann['bbox'])
                    scores.append(ann['score'])
                    labels.append(ann['category_id'])
            boxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        
        #final_boxes, final_scores, final_labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        #final_boxes, final_scores, final_labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        #final_boxes, final_scores, final_labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        final_boxes, final_scores, final_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i]
            bbox = [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]
            det = {"image_id": imgage_id, "bbox": bbox, "score": float(final_scores[i]), "category_id": int(final_labels[i])}
            print(det)
            final_result.append(det)


    with open(args.json_out, 'wt', encoding='UTF-8') as outfile:
        json.dump(final_result, outfile)

main(args)
