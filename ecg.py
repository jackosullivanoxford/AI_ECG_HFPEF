"""A function for training and evaluating ECG models."""

import copy
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch

import config  # Imports the configuration functions and default values
from dataset import ECGDataset  # Imports the custom dataset class we worked on in dataset.py
from train import Trainer, optimizer_and_scheduler  # Imports the training logic and optimizer setup
from model import ECGModel  # Imports the ECG model architecture

def ecg(
        mode="train",  # Sets mode to either "train" or "eval"
        task="H2PEF_imp_excludes",  # CHANGED DEFAULT TASK TO "H2PEF_imp_excludes", THIS SHOULD EXIST IN config.py
        eval_model_path=".",  # Path for evaluating models, used in "eval" mode
        eval_model_name="best.pt",  # Name of the model to evaluate, used in "eval" mode
        use_label_file_for_eval=False,  # Whether to use a label file or evaluate all files
        cfg_updates={},  # Any updates to the config dictionary, applied to config.cfg
        log_path="",  # Path to the log file, for logging results in training or eval
    ):
    """
    A function to train and evaluate ECG models.
    
    Arguments:
        mode: Either "train" or "eval".
        task: The task to train or eval on. Should be a key in config.py's task_cfg.
        eval_model_path: Only used in eval mode. Path to the directory for evaluation.
        eval_model_name: Only used in eval mode. Name of the model to evaluate.
        use_label_file_for_eval: Only used in eval mode. If true, load a label file;
            if false, evaluate on all files in a directory.
        cfg_updates: A nested dict whose schema is a partial copy of config.py's cfg.
            This dict will be used to overwrite the elements in cfg, and to name directories
            during training.
        log_path: Mainly for use with Stanford infrastructure. If set, the file at this path
            will be copied to the model directory after training/evaluation.
    """

    warnings.filterwarnings("ignore")  # Ignore warnings to keep output clean
    torch.manual_seed(0)  # Set random seed for reproducibility in PyTorch
    np.random.seed(0)  # Set random seed for reproducibility in NumPy

    # Update config with task-specific configuration from config.task_cfg and cfg_updates
    cfg = config.update_config_dict(
        config.cfg,  # Base config (from config.py)
        config.task_cfg[task],  # Task-specific updates from task_cfg
    )
    cfg = config.update_config_dict(
        cfg,  # The updated base config after applying task-specific updates
        cfg_updates,  # Any further updates passed in from the user (e.g., hyperparameter tuning)
    )

    # Set the device to GPU (cuda) if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)  # Flush ensures immediate printing of the device info
    
    # Generate a unique name for the model based on the task and config updates (used for saving logs/models)
    model_name = f"{task},{config.dict_to_str(cfg_updates)}"
    print(model_name, flush=True)
    
    if mode == "train":
        # Create output directory to save model checkpoints, logs, etc.
        # CHANGED TO CREATE A SUBFOLDER FOR EACH TASK USING {task} IN THE PATH
        output = os.path.join(cfg["optimizer"]["save_path"], task, model_name)
        os.makedirs(output, exist_ok=True)  # Ensure the directory exists
    else:
        # If in evaluation mode, use the specified evaluation model path
        output = eval_model_path 

    # Initialize the ECG model using the config settings
    model = ECGModel(
        cfg["model"],  # Model-specific parameters from config.py (model_type, conv_width, etc.)
        num_input_channels=(len(cfg["dataloader"]["leads"])
            if cfg["dataloader"]["leads"] else 12),  # Number of input channels, 12 by default
        num_outputs=len(cfg["dataloader"]["label_keys"]),  # Number of outputs based on labels
        binary=cfg["dataloader"]["binary"]  # Whether the task is binary classification
    ).float()

    # If using a GPU with multiple devices, enable data parallelism
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    
    model.to(device)  # Move the model to the appropriate device (GPU/CPU)

    # Initialize the optimizer and learning rate scheduler using settings from config.py
    optim, scheduler = optimizer_and_scheduler(cfg["optimizer"], model)

    # If training or using a label file for evaluation, load the dataset and dataloaders
    if mode == "train" or use_label_file_for_eval:
        # Initialize the dataset for each split (train, valid, test, all)
        datasets = {k: ECGDataset(cfg["dataloader"], k, output=output) for k in ["train", "valid", "test", "all"]}
        
        # Create PyTorch DataLoaders for batching and loading data
        dataloaders = {
            key:
                torch.utils.data.DataLoader(
                    datasets[key],  # Dataset object for each split
                    batch_size=cfg["optimizer"]["batch_size"],  # Batch size from config
                    num_workers=cfg["dataloader"]["n_dataloader_workers"],  # Number of workers for parallel loading
                    shuffle=(key == "train"),  # Shuffle the training data only
                    drop_last=(key == "train"),  # Drop last batch if it's smaller in training
                    pin_memory=True,  # Optimize memory transfer when using a GPU
                )
            for key in ["train", "valid", "test", "all"]}
    else:
        # If evaluating without a label file, set datasets and dataloaders to None
        datasets, dataloaders = None, None

    # Initialize the Trainer object to handle training and evaluation logic
    trainer = Trainer(
                cfg,  # The complete config used by the trainer
                device,  # The device (GPU/CPU) to train or evaluate on
                model,  # The ECG model
                optim,  # The optimizer
                scheduler,  # The learning rate scheduler
                datasets,  # The datasets for training/validation/testing
                dataloaders,  # The PyTorch DataLoader objects
                output,  # The directory to save model checkpoints and logs
    )

    # If in training mode, start training the model
    if mode == "train":
        trainer.train()

    # If in evaluation mode, load the best model and evaluate it
    elif mode == "eval":
        # CHANGED TO LOAD {task}_best.pt INSTEAD OF JUST "best.pt"
        best_epoch, best_score = trainer.try_to_load(f"{task}_best.pt")  # Load the task-specific model checkpoint
        print(f"Best score seen: {best_score:.3f} at epoch {best_epoch}", flush=True)
        
        if use_label_file_for_eval:
            # Evaluate on the test split if using a label file
            trainer.run_eval_on_split("test", report_performance=True)
        else:
            # Evaluate on all files in the specified directory
            trainer.run_eval_on_all()

    # If a log path is provided, copy the log file to the output directory after training/evaluation
    if log_path:
        shutil.copy(log_path, output)

