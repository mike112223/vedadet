# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch

from vedacore.misc import registry
from .base_bbox_coder import BaseBBoxCoder


@registry.register_module('bbox_coder')
class DeltaXWBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, x2) into delta (dx, dw) and
    decodes delta (dx, dw) back to original bbox (x1, x2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0.),
                 target_stds=(1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 2
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               w_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            w_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, w_ratio_clip)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0.), stds=(1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 2)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 2)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 1]) * 0.5
    pw = proposals[..., 1] - proposals[..., 0]

    gx = (gt[..., 0] + gt[..., 1]) * 0.5
    gw = gt[..., 1] - gt[..., 0]

    dx = (gx - px) / pw
    dw = torch.log(gw / pw)
    deltas = torch.stack([dx, dw], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0.),
               stds=(1., 1.),
               max_shape=None,
               w_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 2)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 2 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        w_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 2)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 2)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::2]
    dw = denorm_deltas[:, 1::2]
    max_ratio = np.abs(np.log(w_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 1]) * 0.5).unsqueeze(1).expand_as(dx)
    # Compute width/height of each roi
    pw = (rois[:, 1] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    x2 = gx + gw * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape)
        x2 = x2.clamp(min=0, max=max_shape)
    bboxes = torch.stack([x1, x2], dim=-1).view_as(deltas)
    return bboxes
