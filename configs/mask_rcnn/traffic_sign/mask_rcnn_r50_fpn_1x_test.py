# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn_r50_fpn_poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained=None,
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ', 'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm', 'Hiệu lệnh', )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        classes=classes,
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train_mask_wo_dup.json'),
    val=dict(
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        classes=classes,
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/val_mask.json'),
    test=dict(
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        classes=classes,
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/val_mask.json'))
