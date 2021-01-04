# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from vedacore.misc import registry
from ..bbox import bbox_overlaps1d


@registry.register_module('iou_calculator')
class BboxOverlaps1D(object):
    """2D IoU Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 2, 3]
        assert bboxes2.size(-1) in [0, 2, 3]
        if bboxes2.size(-1) == 3:
            bboxes2 = bboxes2[..., :2]
        if bboxes1.size(-1) == 3:
            bboxes1 = bboxes1[..., :2]
        return bbox_overlaps1d(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
