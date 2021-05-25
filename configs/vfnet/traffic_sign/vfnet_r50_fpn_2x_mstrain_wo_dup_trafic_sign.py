# The new config inherits a base config to highlight the necessary modification
_base_ = '../vfnet_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained=None,
    backbone=dict(frozen_stages=-1),
    bbox_head=dict(num_classes=7),
)

# Modify dataset related settings
dataset_type = 'COCODataset'

classes = ('Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ', 'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm', 'Hiệu lệnh')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True)
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU

    train=dict(
        classes=classes,
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/train_wo_dup.json',
        pipeline= [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1622, 622), (1800, 700)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    ),

    val=dict(
        classes=classes,
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/val.json',
        pipeline= [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1622, 622),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
    ),
    test=dict(
        classes=classes,
        img_prefix='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/images/',
        ann_file='/data2/zalo-ai-2020/za_traffic_2020/data/traffic_train/val.json',
        pipeline= [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1622, 622),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
    ),
)

optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[16, 22])
total_epochs = 24

