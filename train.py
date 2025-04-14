# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:30:39 2025
@author: hp
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import * 
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from torchmetrics import Accuracy
import random
import wandb
from pytorch_lightning.loggers import WandbLogger
from lit_model import Model

def main():
    # Using deterministic execution 
    #declarations variables fixes :
    dir_checkpoint = Path('./checkpoint_folder/')
    dir_data = Path('database_27.hdf5')
    in_channels = 1 
    out_channels = 3 
    batch_size = 8
    epochs = 20

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_checkpoint,
        filename="my_model-{epoch:03d}-{val_loss:.3f}.ckpt",
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # 1 Create the data loader
    #take 20% of the dataset for validation
    PatientsIdTrain, PatientsIdVal = SelectPatientsTrainVal(dir_data, 0.2)
    #If classification type  = true : datagenerator for image classification
    #If classification type  = false : datagenerator for image segmentation 
    train_ds = HDF5Dataset(dir_data, PatientsIdTrain, transform=None, classification_type=False) 
    val_ds = HDF5Dataset(dir_data, PatientsIdVal, transform=None, mode='valid', classification_type=False)
    n_train = len(train_ds)
    n_val = len(val_ds)
    n_classes = 1

    # 2 - Create data loaders
    # params
    loader_params = dict(batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True) 
    train_dl = DataLoader(train_ds, **loader_params)
    val_dl = DataLoader(val_ds, **loader_params)

    #call the model implemented before :
    model = Model(task="segmentation")

    # Create the trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        max_epochs=epochs,
        devices=[0], 
        callbacks=checkpoint_callback,
        logger=WandbLogger()
    )

    # Train the model
    trainer.fit(model, train_dl, val_dl)
    wandb.finish()

if __name__ == '__main__':
    main()
