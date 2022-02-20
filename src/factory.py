from typing import Dict, Any, Tuple
import os

from .model.model import WoofTrainModule
from .dataset.dataset_processing import DatasetProcessing, DataframePreprocessing
from .model.backbones import get_backbone

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np


def get_data_loader(img_paths: np.array, img_labels: np.array, data_path: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    Get batch generator for our data
    """
    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                    std=[0.2814769, 0.226306, 0.20132513])

    dset = DatasetProcessing(
            data_path, 
            img_paths, 
            img_labels,
            transform=transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        )

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True)
    
    return loader


def get_model(checkpoint, backbone) -> pl.LightningModule:
    """
    Get trained model
    """
    backbone = get_backbone(backbone, pretrined=False)
    model = WoofTrainModule(backbone)

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_preprocessing(base_path: str, csv_name: str, label_name: str):
    """
    Get preprocessing module, which split data and encode labels
    """
    return DataframePreprocessing(os.path.join(base_path, csv_name), label_name)


def get_logger(path: str) -> TensorBoardLogger:
    """
    Get logger for our trainer
    """
    return TensorBoardLogger(path)


def get_trainer(backbone: str,
                LR: float,
                df_preprocessing: Dict[str, str], 
                dataset_t: Dict[str, int], 
                dataet_v:  Dict[str, int], 
                logger: Dict[str, str],
                trainer:  Dict[str, Any]) -> Tuple[
                    pl.Trainer, 
                    pl.LightningModule,
                    DataframePreprocessing,
                    DataLoader,
                    DataLoader]:

    """
    Build pytorch lightning trainer
    """

    name = backbone
    backbone = get_backbone(backbone)
    
    model = WoofTrainModule(backbone, LR=LR)

    preprocessing = get_preprocessing(**df_preprocessing)

    train_loader = get_data_loader(*preprocessing.get_train_data(), **dataset_t)
    test_loader = get_data_loader(*preprocessing.get_valid_data(), **dataet_v)

    logger = get_logger(**logger)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="./checkpoints/ResNet/",
        filename= f"{name}-" + "{epoch:02d}-{valid_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback], **trainer)
    return trainer, model, preprocessing, train_loader, test_loader