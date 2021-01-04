from vedacore.misc import registry
from .temporal_infer_engine import TemporalInferEngine


@registry.register_module('engine')
class TemporalValEngine(TemporalInferEngine):

    def __init__(self, model, meshgrid, converter, num_classes, window_size,
                 overlap_ratio, use_sigmoid, test_cfg, eval_metric):
        super().__init__(model, meshgrid, converter, num_classes, window_size,
                         overlap_ratio, use_sigmoid, test_cfg)
        self.eval_metric = eval_metric

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, img, img_metas):
        dets = self.infer(img, img_metas)
        return dets
