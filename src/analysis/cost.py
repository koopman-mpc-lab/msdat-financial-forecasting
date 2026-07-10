import time

import torch

from src.training.trainer import Trainer


def benchmark_model(model, loader, cfg, device=None):
    trainer = Trainer(model, cfg, device)
    params_m = trainer.count_parameters() / 1e6
    t0 = time.perf_counter()
    trainer.train_epoch(loader)
    epoch_time = time.perf_counter() - t0
    latency = trainer.benchmark_inference(loader)
    return {"params_m": params_m, "epoch_time_s": epoch_time, "latency_ms": latency}
