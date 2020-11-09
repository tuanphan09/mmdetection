# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained=None,
    roi_head=dict(
        bbox_head=dict(num_classes=7),
        mask_head=dict(num_classes=7)))

# Modify dataset related settings
dataset_type = 'COCODataset'

classes = ('Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ', 'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm', 'Hiệu lệnh', )

data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU

    train=dict(
        classes=classes,
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train_traffic_sign_dataset.json',
        pipeline=[  # pipeline, this is passed by the train_pipeline created before.
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=False,
                poly2mask=False),
            dict(type='Resize', img_scale=(1622, 622), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
    ),

    val=dict(
        classes=classes,
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train_traffic_sign_dataset.json',
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1622, 622),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
    ),

    test=dict(
        classes=classes,
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train_traffic_sign_dataset.json',
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1622, 622),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
    ),

)

