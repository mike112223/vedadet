# 1. data
dataset_type = 'RawFrameDataset'
data_root = 'data/ssd_thumos14/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size_divisor = 128
window_size = 768
overlap_ratio = 0.5

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations_thumos14_20cls_val.json',
        img_prefix=data_root + 'resized_data_96_160/images/val',
        pipeline=[
            dict(typename='LoadVideoFromRepo',
                 to_float32=True),
            dict(
                typename='LoadAnnotations'),
            dict(typename='OverlapVideoCrop',
                 window_size=window_size,
                 overlap_ratio=overlap_ratio),
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
        ann_file=data_root + 'annotations_thumos14_20cls_test.json',
        img_prefix=data_root + 'resized_data_96_160/images/test',
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

# # 2. model
num_classes = 20
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [1.0]
num_anchors = scales_per_octave * len(ratios)

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet3d',
        depth=50,
        conv_cfg=dict(typename='Conv3d'),
        norm_eval=True,
        # with_pool2=False,
        frozen_stages=3,
        inflate=(
            (1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),  # TODO
    neck=[
        dict(
            typename='CPNet',
            in_channels=2048,
            num_levels=4,
        ),
        dict(
            typename='TFPN',
            in_channels=[2048, 256, 256, 256, 256],
            out_channels=256,
            start_level=0,
            num_outs=5)
    ],
    head=dict(
        typename='TemporalRetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        use_sigmoid=use_sigmoid))

# # 3. engines
meshgrid = dict(
    typename='TemporalBBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='TemporalBBoxBaseAnchor',
        octave_base_scale=2,
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXWBBoxCoder',
    target_means=[.0, .0],
    target_stds=[1.0, 1.0])

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='TemporalBBoxAnchorCriterion',
        num_classes=num_classes,
        meshgrid=meshgrid,
        bbox_coder=bbox_coder,
        loss_cls=dict(
            typename='FocalLoss',
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            typename='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                typename='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(
                    typename='BboxOverlaps1D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    optimizer=dict(
        typename='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001))

## 3.2 val engine
val_engine = dict(
    typename='TemporalValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='TemporalBBoxAnchorConverter',
        num_classes=num_classes,
        bbox_coder=bbox_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    window_size=window_size,
    overlap_ratio=overlap_ratio,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            typename='nms',
            iou_thr=0.5),
        max_per_img=100),
    max_batch=1,
    level=5,
    use_sigmoid=use_sigmoid,
    eval_metric=None)

# 4. hooks
hooks = [
    dict(typename='OptimizerHook'),
    dict(
        typename='StepLrSchedulerHook',
        step=[70, 90]),
    dict(typename='EvalHook'),
    dict(
        typename='SnapshotHook',
        interval=1),
    dict(
        typename='LoggerHook',
        interval=10)]

# 5. work modes
modes = ['train']
max_epochs = 100

# 6. checkpoint
weights = dict(
    filepath='/DATA/home/yanjiazhu/.cache/torch/checkpoints/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de_state.pth')
# optimizer = dict(filepath='workdir/retinanet_mini/epoch_3_optim.pth')
# meta = dict(filepath='workdir/retinanet_mini/epoch_3_meta.pth')

# 7. misc
seed = 1234
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
