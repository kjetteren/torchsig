"""Hybrid Narrowband Trainer for IQ Classification on Narrowband using both GPU and CPU.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TorchSig imports
from torchsig.transforms.target_transforms import DescToClassIndex
from torchsig.transforms.transforms import (
    RandomPhaseShift,
    Normalize,
    ComplexTo2D,
    Compose,
)
from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.datasets.datamodules import NarrowbandDataModule

# PyTorch Lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from torchsig.utils.narrowband_trainer import NarrowbandTrainer
from torchsig.utils.narrowband_trainer import MetricsLogger


class HybridNarrowbandTrainer(NarrowbandTrainer):
    def train(self):
        """
        Trains the model using the prepared data and model.

        If a checkpoint was loaded, continues training (fine-tuning) from the checkpoint.

        Sets up callbacks, initializes the PyTorch Lightning trainer, and
        starts the training process. After training, it plots metrics and
        the confusion matrix.
        """
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=self.checkpoint_dir,
            filename=self.model_name + '-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max',
        )

        # Metrics Logger Callback
        self.metrics_logger = MetricsLogger()

        # Trainer
        self.trainer = Trainer(
            max_epochs=self.num_epochs,
            callbacks=[checkpoint_callback, self.metrics_logger],
            accelerator='auto',
            devices=-1,
            # No need to specify resume_from_checkpoint when using load_from_checkpoint
        )

        # Train
        self.trainer.fit(self.model, self.datamodule)

        # Get the best checkpoint filename base
        self.best_model_path = checkpoint_callback.best_model_path
        self.filename_base = os.path.splitext(os.path.basename(self.best_model_path))[0]

        # Plot metrics
        self.plot_metrics()

        # Plot confusion matrix
        self.plot_confusion_matrix()