"""Common Lightning optimization wrapper for every model architecture."""

from __future__ import annotations

import lightning as L
import torch

from dlcore.hydromodels.model_config import ModelConfig
from dlcore.hydromodels.model_factory import create_model
from dlcore.utils.losses import finite_mse


class RunoffTrainingModule(L.LightningModule):
    def __init__(self, model_config: ModelConfig | dict):
        super().__init__()
        config = (
            ModelConfig.from_dict(model_config)
            if isinstance(model_config, dict)
            else model_config
        )
        config.validate()
        self.model_config = config
        self.model = create_model(config)
        self.save_hyperparameters({"model_config": config.to_dict()})

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(forcing_window, time_features)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        prediction = self(batch["forcing"], batch.get("time_features"))
        loss = finite_mse(prediction, batch["target"])
        if getattr(self, "_trainer", None) is not None:
            self.log(
                f"{stage}_loss",
                loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage != "train",
                batch_size=batch["forcing"].shape[0],
            )
        return loss

    def training_step(self, batch, batch_index: int) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_index: int) -> torch.Tensor:
        return self.shared_step(batch, "validation")

    def test_step(self, batch, batch_index: int) -> torch.Tensor:
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )
