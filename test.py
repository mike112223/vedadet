## train
import torch

from vedacore.misc import Config
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine

cfg = Config.fromfile('configs/trainval/tad/baseline.py')

dataset = build_dataset(cfg.data.train)
dataloader = build_dataloader(
    dataset,
    cfg.data.samples_per_gpu,
    cfg.data.workers_per_gpu,
    dist=False)

engine = build_engine(cfg.train_engine)
engine = MMDataParallel(
    engine.cuda(), device_ids=[torch.cuda.current_device()])

for data in dataloader:
    x = engine(data)
    break

## test
import torch

from vedacore.misc import Config
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine

cfg = Config.fromfile('configs/trainval/tad/baseline.py')

dataset = build_dataset(cfg.data.val, dict(test_mode=True))
dataloader = build_dataloader(
    dataset,
    cfg.data.samples_per_gpu,
    cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)

engine = build_engine(cfg.val_engine)
engine = MMDataParallel(
    engine.cuda(), device_ids=[torch.cuda.current_device()])

for data in dataloader:
    x = engine(data)
    break

## afo train
import torch

from vedacore.misc import Config
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine

cfg = Config.fromfile('configs/trainval/tad/fcos.py')

dataset = build_dataset(cfg.data.train)
dataloader = build_dataloader(
    dataset,
    cfg.data.samples_per_gpu,
    cfg.data.workers_per_gpu,
    dist=False)

engine = build_engine(cfg.train_engine)
engine = MMDataParallel(
    engine.cuda(), device_ids=[torch.cuda.current_device()])

for data in dataloader:
    x = engine(data)
    break
