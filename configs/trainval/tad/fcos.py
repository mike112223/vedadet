# 1. data
dataset_type = 'RawFrameDataset'
data_root = 'data/thumos14/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size_divisor = 128
window_size = 768
overlap_ratio = 0.5

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations_thumos14_20cls_val.json',
        img_prefix=data_root + 'extracted_images/images_10fps_resize_96_160/val',
        pipeline=[
            dict(typename='LoadVideoFromRepo',
                 to_float32=True),
            dict(
                typename='LoadAnnotations'),
            dict(typename='VideoRandomCrop',
                 window_size=window_size),
            dict(
                typename='Normalize3d',
                **img_norm_cfg),
            dict(
                typename='RandomFlip3d'),
            dict(
                typename='Pad3d',
                size_divisor=size_divisor),
            dict(typename='DefaultFormatBundle3d'),
            dict(
                typename='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])]),
    val_samples_per_gpu=1,
    val_workers_per_gpu=1,
    val=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations_thumos14_20cls_val.json',
        img_prefix=data_root + 'extracted_images/images_10fps_resize_96_160/val',
        pipeline=[
            dict(typename='LoadVideoFromRepo',
                 to_float32=True),
            dict(
                typename='VideoMultiScaleFlipAug',
                window_size=window_size,
                flip=False,
                transforms=[
                    dict(typename='OverlapVideoCrop',
                         overlap_ratio=overlap_ratio,
                         test_mode=True),
                    dict(typename='RandomFlip3d'),
                    dict(
                        typename='Normalize3d',
                        **img_norm_cfg),
                    dict(
                        typename='Pad3d',
                        size_divisor=window_size),
                    dict(typename='DefaultFormatBundle3d'),
                    dict(
                        typename='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'window_size'))])])
)

# 2. model
num_classes = 20
strides = [16]
use_sigmoid = True
regress_ranges = [(-1, 10000)]

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='C3D',
        style='pytorch',
        conv_cfg=dict(typename='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(typename='ReLU'),),
    head=dict(
        typename='AFOHead',
        num_classes=num_classes,
        in_channels=512,
        stacked_convs=4,
        feat_channels=256,
        strides=strides,
        use_sigmoid=use_sigmoid,
        conv_cfg=dict(typename='Conv1d'),
        norm_cfg=None))

# 3. engines
meshgrid = dict(
    typename='TemporalPointAnchorMeshGrid',
    strides=strides)

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='TemporalPointAnchorCriterion',
        num_classes=num_classes,
        meshgrid=meshgrid,
        strides=strides,
        regress_ranges=regress_ranges,
        center_sampling=False,
        center_sample_radius=1.5,
        loss_cls=dict(
            typename='CrossEntropyLoss',
            use_sigmoid=use_sigmoid,
            loss_weight=1.0),
        loss_bbox=dict(
            typename='IoULoss',
            loss_weight=1.0)),
    optimizer=dict(
        typename='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        paramwise_cfg=dict(
            bias_lr_mult=2.,
            bias_decay_mult=0.)))

## 3.2 val engine
val_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='PointAnchorConverter',
        num_classes=num_classes,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(typename='nms', iou_thr=0.5),
        max_per_img=100),
    use_sigmoid=use_sigmoid,
    eval_metric=None)

# 4. hooks
hooks = [
    dict(
        typename='OptimizerHook',
        grad_clip=dict(
            max_norm=35,
            norm_type=2)),
    dict(
        typename='StepLrSchedulerHook',
        step=[8, 11],
        warmup='constant',
        warmup_iters=500,
        warmup_ratio=1.0 / 10),
    dict(
        typename='SnapshotHook',
        interval=1),
    dict(
        typename='LoggerHook',
        interval=10),
    dict(typename='EvalHook')]

# 5. work modes
modes = ['train', 'val']
max_epochs = 12

# 6. misc
weights = dict(
    filepath='/DATA/home/yanjiazhu/.cache/torch/checkpoints/c3d_sports1m_pretrain_20201016-dcc47ddc.pth')
# optimizer = dict(filepath='workdir/fcos/epoch_1_optim.pth')
# meta = dict(filepath='workdir/fcos/epoch_1_meta.pth')

# 7. misc
seed = 0
dist_params = dict(backend='nccl')
log_level = 'INFO'
