# split train - val
python3 tools/za_traffic/split_coco_ann.py --having-annotations --train_size 0.85 /data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train_traffic_sign_dataset.json /data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train.json /data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/val.json

python tools/za_traffic/print_distribution.py ../za_traffic_2020/data/traffic_train/val_wo_dup.json

python tools/za_traffic/preprocess.py --in_ann ../za_traffic_2020/data/traffic_train/val.json --out_ann ../za_traffic_2020/data/traffic_train/val_wo_dup.json


# check and train
python tools/print_config.py configs/mask_rcnn/traffic_sign/mask_rcnn_r50_fpn_1x_traffic_sign.py

python tools/train.py configs/retinanet/traffic_sign/retinanet_r50_fpn_1x_traffic_sign.py


# get result
python tools/za_traffic/prepare_ann.py --example ../za_traffic_2020/data/traffic_train/val.json --img_dir ../za_traffic_2020/data/traffic_public_test/images --json_out ../za_traffic_2020/data/traffic_public_test/test.json

python tools/test.py work_dirs/cascade_rcnn_r50_fpn_20e_traffic_sign/test_config.py work_dirs/cascade_rcnn_r50_fpn_20e_traffic_sign/latest.pth --eval bbox # test with ann
python tools/test.py configs/retinanet/traffic_sign/retinanet_r50_fpn_1x_traffic_sign.py work_dirs/retinanet_r50_fpn_1x_traffic_sign/latest.pth --format-only --options "jsonfile_prefix=./tmp/result"

python tools/za_traffic/post_process.py --json_in tmp/result.bbox.json --json_out tmp/final.json



# analysis

python tools/analyze_logs.py plot_curve work_dirs/retinanet_r50_fpn_1x_traffic_sign/20201107_153211.log.json --keys loss_cls loss_bbox --legend cls bbox --out losses.png

python tools/analyze_logs.py plot_curve work_dirs/retinanet_r50_fpn_1x_traffic_sign/20201107_153211.log.json --keys loss_cls loss_bbox --out losses.png

python tools/analyze_logs.py plot_curve work_dirs/retinanet_r50_fpn_1x_traffic_sign_full/20201107_134149.log.json work_dirs/retinanet_r50_fpn_1x_traffic_sign/20201107_153211.log.json --keys bbox_mAP --legend full split --out mAP.png

python tools/analyze_logs.py plot_curve work_dirs/retinanet_r50_fpn_1x_traffic_sign/20201107_153211.log.json --keys bbox_mAP --out mAP.png

python tools/coco_error_analysis.py tmp/retinanet_r50_fpn_1x_traffic_sign.bbox.json tmp --ann ../za_traffic_2020/data/traffic_train/val.json


# visualize bbox

python tools/browse_dataset.py configs/cascade_rcnn/traffic_sign/cascade_rcnn_r50_fpn_1x_traffic_sign.py --output-dir ../za_traffic_2020/data/traffic_train/images_bbox  --not-show # from train ann

python tools/test.py work_dirs/cascade_rcnn_r50_pretrained_fpn_20e_traffic_sign/test_config.py work_dirs/cascade_rcnn_r50_pretrained_fpn_20e_traffic_sign/epoch_19.pth --show-dir tmp/public_test_painted_result # no ann needed
