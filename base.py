from datetime import datetime
import lightning as L
from lightning import LightningModule
import torch
from models.mlp import MultiLayerPerceptron
from typing import Any, Tuple
from data_modules.har import HarDataModule
from abc import ABC, abstractmethod
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse
from pathlib import Path


class Configuration(ABC):
    def __init__(self, *args, **kwargs):
        """This should include common configuration parameters for model, data
        module and trainer.
        """
        pass

    @abstractmethod
    def model(
        self, backbone_ckpt_path: str = None, *args, **kwargs
    ) -> L.LightningModule:
        """This method returns the model to be trained.

        Parameters
        ----------
        backbone_ckpt_path : str, optional
            The path to the backbone checkpoint file (ignored in pretrain models)

        Returns
        -------
        L.LightningModule
            The model to be trained.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def data_module(self, path, *args, **kwargs) -> L.LightningDataModule:
        """
        Abstract method for creating a LightningDataModule.

        Parameters
        ----------
        path : str
            The path to the data.

        Returns
        -------
        L.LightningDataModule
            The created LightningDataModule.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def trainer(
        self,
        save_dir: str,
        name: str,
        version: str,
        epochs: int = 1,
        accelerator: str = "cpu",
        monitor: str = None,
        mode: str = "min",
    ) -> L.Trainer:
        """
        Create and return an instance of the L.Trainer class for training a model.

        Parameters
        ----------
        save_dir : str
            The directory to save the training logs and checkpoints.
        name : str
            The name of the trainer.
        version : str
            The version of the trainer.
        epochs : int, optional
            The number of training epochs, by default 1.
        accelerator : str, optional
            The accelerator to use for training, by default "cpu".
        monitor : str, optional
            The metric to monitor for checkpointing, by default None.
        mode : str, optional
            The mode for the monitored metric (either "min" or "max"), by default "min".

        Returns
        -------
        L.Trainer
            An instance of the L.Trainer class for training a model.
        """
        logger = CSVLogger(
            save_dir=save_dir,
            name=name,
            version=version,
            flush_logs_every_n_steps=1,
        )

        model_checkpoint = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_last=True,
        )

        return L.Trainer(
            accelerator=accelerator,
            max_epochs=epochs,
            log_every_n_steps=10,
            logger=logger,
            callbacks=[model_checkpoint],
        )


def train_model(
    config: Configuration,
    model_kwargs: dict,
    data_module_kwargs: dict,
    trainer_kwargs: dict,
    backbone_ckpt_path: str = None,
    resume_ckpt_path: str = None,
) -> Any:
    """
    Train the model using the provided configuration and return the result.

    Parameters
    ----------
    config : Configuration
        The configuration object that contains the settings for the model, data module, and trainer.
    model_kwargs : dict
        Additional keyword arguments to be passed to the model constructor.
    data_module_kwargs : dict
        Additional keyword arguments to be passed to the data module constructor.
    trainer_kwargs : dict
        Additional keyword arguments to be passed to the trainer constructor.
    backbone_ckpt_path : str, optional
        The path to the checkpoint file for the backbone model, if applicable. Default is None.
    resume_ckpt_path : str, optional
        The path to the checkpoint file to resume training from, if applicable. Default is None.

    Returns
    -------
    Any
        The result of the training process.

    """
    if backbone_ckpt_path is not None:
        model = config.model(backbone_ckpt_path=backbone_ckpt_path, **model_kwargs)
    else:
        model = config.model(**model_kwargs)

    data_module = config.data_module(**data_module_kwargs)
    trainer = config.trainer(**trainer_kwargs)
    result = trainer.fit(model, data_module, ckpt_path=resume_ckpt_path)
    print(
        f"Model trained and saved at: {trainer.checkpoint_callback.best_model_path}"
    )
    return result


def evaluate_model(
    config: Configuration,
    model_kwargs: dict,
    data_module_kwargs: dict,
    trainer_kwargs: dict,
    resume_ckpt_path: str = None,
) -> Any:
    """
    Evaluate the model using the provided configuration and arguments.

    Parameters
    ----------
    config : Configuration
        The configuration object that specifies the model, data module, and trainer.
    model_kwargs : dict
        The keyword arguments to be passed to the model constructor.
    data_module_kwargs : dict
        The keyword arguments to be passed to the data module constructor.
    trainer_kwargs : dict
        The keyword arguments to be passed to the trainer constructor.
    resume_ckpt_path : str, optional
        The path to a checkpoint file to resume training from, by default None.

    Returns
    -------
    Any
        The result of the model evaluation.

    """
    model = config.model(**model_kwargs)
    data_module = config.data_module(**data_module_kwargs)
    trainer = config.trainer(**trainer_kwargs)
    return trainer.test(
        model, datamodule=data_module, ckpt_path=resume_ckpt_path
    )
