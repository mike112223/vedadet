import argparse

import torch
import numpy as np

from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--fps', type=int, default=25, help='fps')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--iou', type=float, default=0.5, nargs='+', help='iou threshold')
    parser.add_argument('--proposal_nums', type=int, default=100, nargs='+', help='proposal nums')

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = engine.to(device)

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader


def test(engine, data_loader, fps):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    results_dict = {}
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = engine(data)[0]

        result = np.concatenate(result, axis=0)

        results.append(result)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()

        name = data['img_metas'][0].data[0][0]['ori_filename']
        results_dict[name] = proposal_format(result, fps)

        # import pdb
        # pdb.set_trace()

    return results, results_dict


def proposal_format(result, fps):
    '''
        {'score': 0.0021732747554779053,
        'segment': [145.11466666666664, 148.30399999999997]},
    '''
    results = []
    for i in range(result.shape[0]):
        x1, _, x2, _, score = result[i]
        results.append({'score': score, 'segment': [x1 / fps, x2 / fps]})
    return results


def result_format(result):
    '''
        dict_keys(['version', 'results', 'external_data'])
    '''
    results = {'version': '', 'external_data':{}}
    results['results'] = result
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.json')):
        raise ValueError('The output file must be a json file.')

    engine, data_loader = prepare(cfg, args.checkpoint)

    results, results_dict = test(engine, data_loader, args.fps)

    data_loader.dataset.evaluate(results, 'recall', iou_thr=args.iou, proposal_nums=args.proposal_nums)

    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(result_format(results_dict), args.out)


if __name__ == '__main__':
    main()


'''

'''
