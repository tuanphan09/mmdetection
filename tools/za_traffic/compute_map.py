import os
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

parser = argparse.ArgumentParser(description='Compute mAP')

parser.add_argument('--gt', dest='gt', type=str, required=True,
                    help="Ann ground truth")

parser.add_argument('--det', dest='det', type=str, required=True,
                    help="Ann dectection")

args = parser.parse_args()


annType = 'bbox'

cocoGt = COCO(args.gt)
cocoDt = cocoGt.loadRes(args.det)

imgIds = sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
