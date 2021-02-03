import argparse

import cv2
import torch
import numpy as np

from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine
from vedadet.misc.bbox import pseudo_bbox


COLOR = {
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [0, 255, 255],
    4: [255, 255, 0],
    5: [255, 0, 255],
    6: [127, 0, 0],
    7: [0, 127, 0],
    8: [0, 0, 127],
    9: [127, 127, 0],
    10: [0, 127, 127],
    11: [127, 0, 127],
    12: [255, 127, 0],
    13: [127, 255, 0],
    14: [255, 0, 127],
    15: [127, 0, 255],
    16: [0, 255, 127],
    17: [0, 127, 255],
    18: [67, 0, 0],
    19: [0, 67, 0],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show_th', default=0.05, type=float)
    parser.add_argument('--out', help='output result file in pickle format')

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader


def plot(ann, data, result, th, idx, stride):
    length = data['duration']
    res_img = np.ones((50, length, 3), dtype=np.uint8) * 255
    labels = ann['labels']
    bboxes = ann['bboxes']

    for i in range(len(result)):
        res = result[i]
        dt_mask = res[:, 4] > th
        gt_mask = labels == i

        dt_left = res[dt_mask]
        gt_left = bboxes[gt_mask]

        for j in range(len(dt_left)):
            x1, y1, x2, y2, s = dt_left[j]
            cv2.rectangle(res_img, (int(x1), int(y1)), (int(x2), int(y2) + 19), COLOR[i], 2)

        for j in range(len(gt_left)):
            x1, y1, x2, y2 = gt_left[j]
            cv2.rectangle(res_img, (int(x1), int(y1) + 30), (int(x2), int(y2) + 47), COLOR[i], 2)

    clip = 0
    while clip < length:
        cv2.line(res_img, (int(clip), 0), (int(clip), 50), [0, 0, 0])
        clip += stride

    cv2.imwrite(f'output/res{idx}.jpg', res_img)


def test(engine, data_loader, th, stride):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = engine(data)[0]

        results.append(result)
        batch_size = len(data['img_metas'][0].data)

        ann = dataset.get_ann_info(i)
        data_info = dataset.data_infos[i]
        ann['bboxes'] = pseudo_bbox(
            ann['bboxes'], mode='numpy')
        ann['bboxes_ignore'] = pseudo_bbox(
            ann['bboxes_ignore'], mode='numpy')

        plot(ann, data_info, result, th, i, stride)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    stride = cfg.window_size * cfg.overlap_ratio

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader, args.show_th, stride)

    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(results, args.out)

    data_loader.dataset.evaluate(results)

if __name__ == '__main__':
    main()
