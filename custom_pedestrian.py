
# Inherit and overwrite part of the config based on this config
# _base_ = 'mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py'
_base_ = 'mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'

data_root = 'D:/Side/2024_Sejoong_Jaywalking/DB/detection/' # dataset root

train_batch_size_per_gpu = 2
train_num_workers = 1

max_epochs = 500
stage2_num_epochs = 250
base_lr = 0.00025


metainfo = {
    'classes': ('jaywalking pedestrian', 'crosswalk pedestrian',),
    'palette': [(255, 0, 0), (0, 0, 255)]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/images/'),
        ann_file='train/annotations-2.json'))

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='test/images/'),
        ann_file='test/annotations-2.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'test/annotations-2.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=2))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 50 to 100 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=200,  # only keep latest 2 checkpoints
        save_best='auto',
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load COCO pre-trained weight
load_from = './detection_ckp/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
# load_from = 'D:/Side/2024_Sejoong_Jaywalking/backup/detection/rtmdet_tiny_1xb4-20e_pedestrian/label-1/best_coco_bbox_mAP_epoch_16.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
