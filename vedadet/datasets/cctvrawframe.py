
import numpy as np
import json

from .custom import CustomDataset
from vedacore.misc import registry
from vedadet.misc.evaluation import eval_map, eval_recalls
from vedadet.misc.bbox import pseudo_bbox


@registry.register_module('dataset')
class CCTVRawFrameDataset(CustomDataset):

    CLASSES = ('Embrace', 'ShakeHands', 'Meeting', 'ReleaseConference',
               'Conference', 'Photograph', 'TakeOffPlane',
               'MilitaryParade', 'MilitaryExercise', 'RocketLaunching',
               'ConferenceSpeech', 'Interview')

    def __init__(self, fps=10, **kwargs):
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.fps = fps
        super(CCTVRawFrameDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        json_file = json.load(open(ann_file, 'r'))
        data_infos = []
        for idx, (key, value) in enumerate(json_file.items()):
            data_infos.append(
                dict(id=idx, filename=key,
                     ann=value['annotations'],
                     duration=int(value['duration_second'] * self.fps),
                     width=240, height=180))
        return data_infos

    def get_ann_info(self, idx):
        data_info = self.data_infos[idx]
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for ann in data_info['ann']:
            bbox = [coor * self.fps for coor in ann['segment']]
            label = self.cat2label[ann['label']]
            gt_bboxes.append(bbox)
            gt_labels.append(label)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 2), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 2), dtype=np.float32)

        assert(len(gt_bboxes) > 0)

        ann = dict(
            bboxes=gt_bboxes.astype(np.float32),
            labels=gt_labels.astype(np.int64),
            bboxes_ignore=gt_bboxes_ignore.astype(np.float32))

        return ann

    def __len__(self):
        return len(self.data_infos)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 jsonfile_prefix=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=[(0, 5.66), (5.66, 9.79), (9.79, 100), (0, 100)]):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if jsonfile_prefix is not None:
            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix)

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = []
        for i in range(len(self)):
            ann = self.get_ann_info(i)
            ann['bboxes'] = pseudo_bbox(
                ann['bboxes'], mode='numpy')
            ann['bboxes_ignore'] = pseudo_bbox(
                ann['bboxes_ignore'], mode='numpy')
            annotations.append(ann)

        # import pdb
        # pdb.set_trace()

        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            # results : len:pics, results[0][0] shape (n,5)
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
