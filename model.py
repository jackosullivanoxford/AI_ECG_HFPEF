"""Objects for defining AI-ECG models."""

import torch
from torch import nn
import torchvision
import numpy as np
import sklearn.metrics
import scipy.special
from torchinfo import summary

from model_specs import specs, two_d_specs  # Import model specifications for 1D and 2D architectures


class ECGModel(torch.nn.Module):
    """
    A convolutional neural network for ingesting ECG data.

    Arguments:
        cfg: The config dict, like config.py's cfg["model"].
        num_input_channels: The number of channels (ECG leads).
        num_outputs: The number of model outputs.
        binary: Whether the output of the model is one or more binary predictions.
        score: Optionally, a score function to rank models by, taking (y, yh, loss)
            as input.

    This object builds 1- and 2-d backbones based on model_specs.py, where other 
    architectures can be easily implemented.
    """

    def __init__(self, cfg, num_input_channels, num_outputs, binary, score=None):
        super(ECGModel, self).__init__()
        self.cfg = cfg  # Store the configuration for the model
        self.out_channels = num_outputs  # Number of output channels for the model
        self.binary = binary  # Is the output binary classification or regression?
        self.num_channels_for_adaptive_2d = 12  # Number of input channels for 2D conv layers (for adaptive 2D models)

        # Choose between 1D and 2D model architecture based on the configuration
        if self.cfg["is_2d"]:
            self.num_input_channels = 1  # For 2D models, input channels are set to 1 (image-like inputs)
            self.features, in_channels = self.make_2d_conv()  # Build the 2D convolution layers
        else:
            self.num_input_channels = num_input_channels  # Use the given number of input channels for 1D models
            self.features, in_channels = self.make_conv()  # Build the 1D convolution layers

        self.classifier = self.make_fc(in_channels)  # Create fully connected (FC) layers for classification

        # Define loss function and scoring method based on whether the task is binary classification or regression
        if self.binary:
            self.score = score or (lambda y, yh, loss: sklearn.metrics.roc_auc_score(y, yh))  # Use AUC score for binary tasks
            w = torch.from_numpy(np.array([cfg["pos_weight"]])).float()  # Handle class imbalance using pos_weight
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=w)  # Binary cross entropy loss with logits
        else:
            self.score = score or (lambda y, yh, loss: sklearn.metrics.r2_score(y, yh))  # Use R2 score for regression tasks
            self.loss = torch.nn.MSELoss()  # Mean Squared Error (MSE) loss for regression

        self.float()  # Ensure model uses float precision

        # Print model structure and number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self, flush=True)
        print(f"num params: {num_params}", flush=True)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        if self.cfg["is_2d"]:
            x = x.unsqueeze(1)  # For 2D models, add a channel dimension
        x = self.features(x)  # Apply convolutional layers
        x = torch.flatten(x, 1)  # Flatten the output for fully connected layers
        x = self.classifier(x)  # Apply fully connected layers
        return x

    def train_step(self, x, y):
        """
        Performs a single training step: forward pass and computes loss.
        """
        y_hat = self.forward(x)  # Forward pass
        loss = self.loss(y_hat, y)  # Compute loss
        return (y, y_hat, loss)  # Return ground truth, prediction, and loss

    def make_conv(self):
        """
        Creates a 1D convolutional backbone based on model specifications.
        """
        spec = specs[self.cfg["model_type"]][0]  # Get the model specification for the selected model type
        in_channels = self.num_input_channels  # Initialize the input channels for the convolution layers

        layers = []  # List to store the layers of the model
        for d in spec:
            if type(d) == int:
                d = ("C", d, self.cfg["conv_width"], 1)  # Convert single integers into convolutional layer definitions
            elif type(d) == str:
                d = (d,)

            # Convolutional layer
            if d[0] == "C":
                self.check_length(d, [2, 3, 4])
                if len(d) == 2:
                    d = (d[0], d[1], self.cfg["conv_width"], 1)
                if len(d) == 3:
                    d = (d[0], d[1], d[2], 1)
                conv = nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=d[3])  # Create Conv1d layer
                if self.cfg["batch_norm"]:
                    layers += [conv, nn.BatchNorm1d(d[1]), nn.ReLU(inplace=False)]  # Add batch normalization and ReLU
                else:
                    layers += [conv, nn.ReLU(inplace=False)]  # Without batch normalization
                in_channels = d[1]

            # Residual block
            elif d[0] == "B":
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], False)
                layers += [Block(in_channels, d[1], d[2], batch_norm=self.cfg["batch_norm"])]  # Add residual block
                in_channels = d[1]

            # Residual layer with multiple blocks
            elif d[0] == "L":
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], 1)
                layers += self.make_layer(in_channels, d[1], d[2], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                in_channels = d[1]

            # Full ResNet core
            elif d[0] == "R":
                self.check_length(d, [2])
                nbs = d[1]
                layers += self.make_layer(in_channels, 64, nbs[0], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(64, 128, nbs[1], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(128, 256, nbs[2], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(256, 512, nbs[3], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                in_channels = 512

            # Max pooling
            elif d[0] == "m":
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 2)
                layers += [nn.MaxPool1d(kernel_size=d[1], stride=d[1])]

            # Single conv layer
            elif d[0] == "c":
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], self.cfg["conv_width"])
                layers += [nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=1)]
                in_channels = d[1]

            # Batch normalization
            elif d[0] == "b":
                if self.cfg["batch_norm"]:
                    self.check_length(d, [1])
                    layers += [nn.BatchNorm1d(in_channels)]

            # ReLU activation
            elif d[0] == "r":
                self.check_length(d, [1])
                layers += [nn.ReLU(inplace=False)]

            # Adaptive average pooling
            elif d[0] == "a":
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 1)
                layers += [nn.AdaptiveAvgPool1d(output_size=d[1])]
                in_channels = d[1] * in_channels

            elif d[0] == "d":
                layers += [nn.Dropout(self.cfg["drop_prob"])]
            else:
                raise NotImplementedError(d)

        return nn.Sequential(*layers), in_channels  # Return the sequential model and final output channels

    def make_2d_conv(self):
        """
        Creates a 2D convolutional backbone based on model specifications.
        """
        spec = two_d_specs[self.cfg["model_type"]][0]  # Get the model spec for 2D architectures
        in_channels = self.num_input_channels  # Initialize input channels

        layers = []
        for d in spec:
            if type(d) == str:
                d = (d,)
            elif type(d) == int:
                d = ("C", int(self.cfg["conv_width"]), (1, 1))

            # Convolutional layer
            if d[0] == "C":
                if len(d) == 3:
                    d = (d[0], d[1], (d[2], d[2]), (1, 1))
                conv = nn.Conv2d(in_channels, d[1], kernel_size=d[2], padding=d[3])
                if self.cfg["batch_norm"]:
                    layers += [conv, nn.BatchNorm2d(d[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = d[1]

            # Max pooling
            elif d[0] == "m":
                if len(d) == 1:
                    d = (d[0], (2, 2))
                layers += [nn.MaxPool2d(kernel_size=d[1], stride=d[1])]

            # Adaptive average pooling
            elif d[0] == "a":
                if len(d) == 1:
                    d = (d[0], (3, 3))
                layers += [nn.AdaptiveAvgPool2d(output_size=d[1])]
                in_channels = np.prod(d[1]) * in_channels

            elif d[0] == "d":
                layers += [nn.Dropout(self.cfg["drop_prob"])]
            else:
                raise NotImplementedError(d)

        return nn.Sequential(*layers), in_channels

    def make_fc(self, in_channels):
        """
        Creates fully connected layers for classification.
        """
        if self.cfg["is_2d"]:
            spec = two_d_specs[self.cfg["model_type"]][1]
        else:
            spec = specs[self.cfg["model_type"]][1]

        spec = spec.copy()  # Copy the FC layer spec
        spec.append(self.out_channels)  # Append the output channels

        layers = []
        for d in spec:
            if type(d) == int:
                layers.append(nn.Linear(in_channels, d))
                in_channels = d
            elif d == "r":
                layers.append(nn.ReLU(inplace=True))
            elif d == "d":
                layers.append(nn.Dropout(self.cfg["drop_prob"]))
            elif d == "b":
                layers.append(nn.BatchNorm1d(in_channels))
            elif type(d) == tuple and d[0] == "d":
                layers.append(nn.Dropout(d[1]))
            else:
                raise NotImplementedError(d)
        return nn.Sequential(*layers)

    def check_length(self, obj, lengths):
        """
        Helper function to ensure layers are properly defined.
        """
        assert len(obj) in lengths, "{} is malformed".format(obj)

    def make_layer(self, in_channels, out_channels, n_blocks, kernel_size=3, block=None, batch_norm=False):
        """
        Creates a residual layer with multiple blocks.
        """
        if block is None:
            block = Block
        blocks = [block(in_channels, out_channels, average_pool=True, kernel_size=kernel_size, batch_norm=batch_norm)]
        blocks += [block(out_channels, out_channels, kernel_size=kernel_size, batch_norm=batch_norm) for _ in range(1, n_blocks)]
        return nn.Sequential(*blocks)


class Block(nn.Module):
    """
    A basic block for the ECG ResNet architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, average_pool=False, batch_norm=True):
        super(Block, self).__init__()
        if batch_norm:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, padding=1, kernel_size=kernel_size, stride=(2 if average_pool else 1)),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(out_channels, out_channels, padding=1, kernel_size=kernel_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, padding=1, kernel_size=kernel_size, stride=(2 if average_pool else 1)),
                nn.ReLU(inplace=False),
                nn.Conv1d(out_channels, out_channels, padding=1, kernel_size=kernel_size),
                nn.ReLU(inplace=False)
            )

        self.downsample = None
        if average_pool:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
        x = identity.clone() + self.backbone(x)
        return x


class BottleneckBlock(nn.Module):
    """
    A bottleneck block for the ECG ResNet architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, average_pool=False, batch_norm=False):
        super(BottleneckBlock, self).__init__()
        middle_channels = int(out_channels / 4)
        if batch_norm:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, middle_channels, padding=0, kernel_size=1, stride=(2 if average_pool else 1)),
                nn.BatchNorm1d(middle_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels, middle_channels, padding=1, kernel_size=kernel_size),
                nn.BatchNorm1d(middle_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels, out_channels, padding=0, kernel_size=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, middle_channels, padding=0, kernel_size=1, stride=(2 if average_pool else 1)),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels, middle_channels, padding=1, kernel_size=kernel_size),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels, out_channels, padding=0, kernel_size=1),
                nn.ReLU(inplace=False)
            )

        self.downsample = None
        if average_pool:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
        x = identity.clone() + self.backbone(x)
        return x

