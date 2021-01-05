import math

import torch

from vedacore.misc import registry
from vedadet.bridge import build_converter, build_meshgrid
from vedadet.misc.bbox import bbox2result, multiclass_nms, pseudo_bbox
from .base_engine import BaseEngine


@registry.register_module('engine')
class TemporalInferEngine(BaseEngine):

    def __init__(self, model, meshgrid, converter, num_classes, window_size,
                 overlap_ratio, use_sigmoid, test_cfg, max_batch, level):
        super().__init__(model)
        self.meshgrid = build_meshgrid(meshgrid)
        self.converter = build_converter(converter)
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap_ratio))
        self.test_cfg = test_cfg
        self.max_batch = max_batch
        self.level = level

    def extract_feats(self, img):
        feats = self.model(img, train=False)
        return feats

    # def extract_feats(self, img):
    #     n, c, t, h, w = img.shape
    #     img = img.permute(0, 2, 1, 3, 4).reshape(
    #         -1, self.window_size, c, h, w).permute(0, 2, 1, 3, 4)
    #     print(img.shape)

    #     cls_scores, bbox_preds = [], []
    #     for i in range(math.ceil(img.shape[0] / self.max_batch)):
    #         cls_score, bbox_pred = self.model(
    #             img[i * self.max_batch:(i + 1) * self.max_batch], train=False)

    #         for c, b in zip(cls_score, bbox_pred):
    #             cls_scores.append(c)
    #             bbox_preds.append(b)

    #     cls_scores = [torch.cat(cls_scores[i::self.level]) for i in range(self.level)]
    #     bbox_preds = [torch.cat(bbox_preds[i::self.level]) for i in range(self.level)]

    #     return cls_scores, bbox_preds

    def _get_raw_dets(self, img, img_metas):
        """
        Args:
            img(torch.Tensor): shape N*3*H*W, N is batch size
            img_metas(list): len(img_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*5
        """

        feats = self.extract_feats(img)

        featmap_sizes = [feat.shape[-1] for feat in feats[0]]
        dtype = feats[0][0].dtype
        device = feats[0][0].device
        anchor_mesh = self.meshgrid.gen_anchor_mesh(featmap_sizes, img_metas,
                                                    dtype, device)
        # bboxes, scores, score_factor
        dets = self.converter.get_bboxes(anchor_mesh, img_metas, *feats)

        return dets

    def _simple_infer(self, img, img_metas):
        """
        Args:
            img(torch.Tensor): shape N*3*H*W, N is batch size
            img_metas(list): len(img_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*5
        """
        n, c, t, h, w = img.shape
        img = img.permute(0, 2, 1, 3, 4).reshape(
            -1, self.window_size, c, h, w).permute(0, 2, 1, 3, 4)
        print(img.shape, math.ceil(img.shape[0] / self.max_batch))
        dets = []
        for i in range(math.ceil(img.shape[0] / self.max_batch)):
            clip_img = img[i * self.max_batch:(i + 1) * self.max_batch]
            det = self._get_raw_dets(clip_img, img_metas)

        dets.extend(det)
        batch_size = len(dets)

        result_list = []
        bboxes = []
        scores = []
        centernesss = []

        for ii in range(batch_size):
            bbox, score, centerness = dets[ii]
            bboxes.append(bbox + self.stride * ii)
            scores.append(score)
            centernesss.append(centerness)

        bboxes = torch.cat(bboxes)
        scores = torch.cat(scores)
        centernesss = torch.cat(centernesss)

        bboxes = bboxes.clamp(min=0, max=img_metas[0]['img_shape'][0])
        # pseudo bbox [x1, y1, x2, y2]
        bboxes = pseudo_bbox(bboxes)

        det_bboxes, det_labels = multiclass_nms(
            bboxes,
            scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=centernesss)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.cls_out_channels)
        result_list.append(bbox_result)

        return result_list

    def _aug_infer(self, img_list, img_metas_list):
        # TODO
        assert len(img_list) == len(img_metas_list)
        dets = []
        ntransforms = len(img_list)
        for idx in range(len(img_list)):
            img = img_list[idx]
            img_metas = img_metas_list[idx]
            tdets = self._get_raw_dets(img, img_metas)
            dets.append(tdets)
        batch_size = len(dets[0])
        nclasses = len(dets[0][0])
        merged_dets = []
        for ii in range(batch_size):
            single_image = []
            for kk in range(nclasses):
                single_class = []
                for jj in range(ntransforms):
                    single_class.append(dets[jj][ii][kk])
                single_image.append(torch.cat(single_class, axis=0))
            merged_dets.append(single_image)

        result_list = []
        for ii in range(batch_size):
            bboxes, scores, centerness = merged_dets[ii]
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                self.test_cfg.score_thr,
                self.test_cfg.nms,
                self.test_cfg.max_per_img,
                score_factors=centerness)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.cls_out_channels)
            result_list.append(bbox_result)

        return result_list

    def infer(self, img, img_metas):
        if len(img) == 1:
            return self._simple_infer(img[0], img_metas[0])
        else:
            return self._aug_infer(img, img_metas)
