"""Lightning data module for non-overlapping hydrological periods."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader

from dlcore.data_pipeline.window_dataset import ScalerStatistics, WindowDataset


class DLDataModule(L.LightningDataModule):
    def __init__(self, series, scaler: ScalerStatistics, params):
        super().__init__()
        self.series = series
        self.scaler = scaler
        self.params = params
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage=None) -> None:
        if self.train_dataset is None:
            self.train_dataset = WindowDataset(
                self.series,
                self.scaler,
                self.params.train_start,
                self.params.train_end,
                self.params.lookback,
            )
            self.validation_dataset = WindowDataset(
                self.series,
                self.scaler,
                self.params.validation_start,
                self.params.validation_end,
                self.params.lookback,
            )
            self.test_dataset = WindowDataset(
                self.series,
                self.scaler,
                self.params.test_start,
                self.params.test_end,
                self.params.lookback,
            )

    def make_loader(
        self,
        dataset,
        shuffle: bool,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ) -> DataLoader:
        use_pin_memory = (
            self.params.device != "cpu" and torch.cuda.is_available()
            if pin_memory is None
            else pin_memory
        )
        return DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=bool(num_workers > 0),
        )

    def train_dataloader(self) -> DataLoader:
        return self.make_loader(self.train_dataset, True, self.params.num_workers)

    def train_evaluation_dataloader(self) -> DataLoader:
        return self.make_loader(self.train_dataset, False, self.params.num_workers)

    def val_dataloader(self) -> DataLoader:
        return self.make_loader(self.validation_dataset, False, self.params.num_workers)

    def test_dataloader(self) -> DataLoader:
        return self.make_loader(self.test_dataset, False, self.params.num_workers)
