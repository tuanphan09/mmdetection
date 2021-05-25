import json
import argparse
import math

import cv2
import numpy as np


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')

parser.add_argument('--in_ann', dest='in_ann', type=str, required=True,
                    help="Path to read COCO annotations file.")

parser.add_argument('--out_ann', dest='out_ann', type=str, required=True,
                    help="Path to write COCO annotations file.")

parser.add_argument('--rm_dup_bbox', dest='rm_dup_bbox', type=str, default='false', choices=['true', 'false'],
                    help="True if remove duplicated bbox")

parser.add_argument('--expand_bbox', dest='expand_bbox', type=str, default='false', choices=['true', 'false'],
                    help="True if expand bbox down")

parser.add_argument('--rm_tiny_bbox', dest='rm_tiny_bbox', type=str, default='false', choices=['true', 'false'],
                    help="True if remove small bbox")

parser.add_argument('--gen_mask', dest='gen_mask', type=str, default='false', choices=['true', 'false'],
                    help="True if create mask")

args = parser.parse_args()

def save_coco(file, info, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def filter_duplicated_bbox(annotations):
    new_anntations = []
    for ann in annotations:
        ok = True
        for new_ann in new_anntations:
            if new_ann['image_id'] == ann['image_id'] and \
                    new_ann['category_id'] == ann['category_id'] and \
                    IoU(new_ann['bbox'], ann['bbox']) > 0.7:
                ok = False
        
        if ok:
            new_anntations.append(ann)
    return new_anntations

def main(args):
    max_width = 1622
    max_height = 626

    with open(args.in_ann, 'rt', encoding='UTF-8') as in_json:
        coco = json.load(in_json)
        info = coco['info']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']
        
        if args.rm_tiny_bbox == 'true':
            new_anntations = []
            for ann in annotations:
                x, y, w, h = ann['bbox']
                if w * h > 16:
                    new_anntations.append(ann)                    
            print("Removed {} tiny bboxes".format(len(annotations) - len(new_anntations)))
            annotations = new_anntations

        if args.rm_dup_bbox == 'true':
            print("Filter duplicated bboxes")
            ori_bbox = len(annotations)
            annotations = filter_duplicated_bbox(annotations)
            print("Removed {} duplicated bboxes".format(ori_bbox - len(annotations)))
        
        if args.expand_bbox == "true":
            print("Expand bboxes")
            for ann in annotations:
                x, y, w, h = ann['bbox']
                if w < 17:
                    if h < w:
                        rate = max(1.1, min(1.12, w/h))
                    else:
                        rate = 1.08
                elif w < 30:
                    if h < w:
                        rate = max(1.07, min(0.9, w/h))
                    else:
                        rate = 1.05
                elif w < 80:
                    if h < w:
                        rate = 1.04
                    else:
                        rate = 1.03
                else:
                    rate = 1
                
                h = int(rate * h + 0.5)

                if x + w > max_width:
                    w = max_width - x
                if y + h > max_height:
                    h = max_height - y
                
                ann['bbox'] = [x, y, w, h]
        
        if args.gen_mask == "true":
            print("Generate masks")
            for ann in annotations:
                x, y, w, h = ann['bbox']
                cat_id = ann['category_id']

                seg = []
                if cat_id == 6: # triangle
                    top_x, top_y = int(x + w / 2), int(y + 0.1 * h)
                    left_x, left_y = int(x + 0.05 * w), int(y + 0.95 * h)
                    right_x, right_y = int(x + 0.95 * w), int(y + 0.95 * h)
                    seg = [top_x, top_y, left_x, left_y, right_x, right_y]
                else: # circle --> ellipse: (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1
                    center_x, center_y = int(x + 0.5 * w), int(y + 0.53 * h)
                    ellipse_width = max(1, int(0.93 * min(center_x - x, x + w - center_x)))
                    ellipse_height = max(1, int(0.91 * min(center_y - y, y + h - center_y)))
                    
                    top_point = []
                    bottom_point = []
                    for point_x in range(center_x - ellipse_width, center_x + ellipse_width, 2):
                        distance = math.sqrt((1 - (point_x - center_x)**2 / (ellipse_width**2)) * (ellipse_height**2)) # = ||y - center_y||
                        distance = int(distance)
                        top_point.append((point_x, center_y - distance))
                        if distance != 0:
                            bottom_point.append((point_x, center_y + distance))
                    
                    for point in top_point: # with x increase
                        seg.append(point[0])
                        seg.append(point[1])
                    
                    for idx in range(len(bottom_point)-1, -1, -1): # with x decrease
                        point = bottom_point[idx]
                        seg.append(point[0])
                        seg.append(point[1])


                ann['segmentation']=[seg]

                # img = cv2.imread('data/validation/' + str(ann['image_id']) + '.png')
                # pts = []
                # for idx in range(0, len(seg), 2):
                #     pts.append([seg[idx], seg[idx+1]])
                # pts = np.array(pts, np.int32)
                # pts = pts.reshape((-1,1,2))

                
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 20, 200), 1)
                # cv2.polylines(img,[pts],True,(0,255,255), 1)
                # cv2.imshow('Mask', img)
                # cv2.waitKey(0)
                
        save_coco(args.out_ann, info, images, annotations, categories)

main(args)


# classes=('Cấm ngược chiều': 0, 'Cấm dừng và đỗ': 0, 'Cấm rẽ': 0,
#                  'Giới hạn tốc độ': 0, 'Cấm còn lại': 0, 'Nguy hiểm':3, 'Hiệu lệnh': 0)
