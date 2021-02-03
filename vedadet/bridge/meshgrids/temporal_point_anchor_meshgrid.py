import torch

from vedacore.misc import registry
from .base_meshgrid import BaseMeshGrid


@registry.register_module('meshgrid')
class TemporalPointAnchorMeshGrid(BaseMeshGrid):

    def __init__(self, strides):
        super().__init__(strides)

    def gen_anchor_mesh(self,
                        featmap_sizes,
                        img_metas,
                        dtype=torch.float,
                        device='cuda'):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas: Nonsense here
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._gen_anchor_mesh_single(featmap_sizes[i], self.strides[i],
                                             dtype, device))
        return mlvl_points

    def _gen_anchor_mesh_single(self, featmap_size, stride, dtype, device):
        """Get points according to feature map sizes."""

        feat_t = featmap_size
        t_range = torch.arange(feat_t, dtype=dtype, device=device)
        points = t_range.reshape(-1, 1) * stride + stride // 2
        return points
