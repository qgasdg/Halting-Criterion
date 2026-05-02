"""학습 중 항상 켜두는 가벼운 디버깅 콜백."""

import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


class GradientHealthCallback(pl.Callback):
    """Optimizer step 직전마다 gradient L2 norm 과 NaN/Inf 를 체크.

    - `debug/grad_norm`: PL 의 `log_every_n_steps` cadence 를 따름 (기본 평균 집계).
    - `debug/grad_nan_param_count`, `debug/grad_inf_param_count`,
      `debug/grad_norm_at_anomaly`: NaN/Inf 가 한 번이라도 감지되면 cadence 와 무관하게
      모든 logger 에 즉시 push 됨.
    - 추가로 `rank_zero_warn` 으로 콘솔 1회 경고.
    """

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        total_norm_sq = 0.0
        nan_params = 0
        inf_params = 0

        for param in pl_module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if torch.isnan(grad).any():
                nan_params += 1
            if torch.isinf(grad).any():
                inf_params += 1
            total_norm_sq += grad.float().pow(2).sum().item()

        grad_norm = math.sqrt(total_norm_sq)
        pl_module.log(
            "debug/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        if nan_params == 0 and inf_params == 0:
            return

        payload = {
            "debug/grad_nan_param_count": float(nan_params),
            "debug/grad_inf_param_count": float(inf_params),
            "debug/grad_norm_at_anomaly": grad_norm,
        }
        for logger in trainer.loggers:
            logger.log_metrics(payload, step=trainer.global_step)

        rank_zero_warn(
            f"[GradientHealth] step={trainer.global_step} "
            f"nan_params={nan_params} inf_params={inf_params} "
            f"grad_norm={grad_norm:.4g}"
        )
