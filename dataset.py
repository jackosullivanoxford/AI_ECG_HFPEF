"""A dataset for loading ECG data."""

import copy
import os
import glob
import tqdm

import numpy as np
import pandas as pd
import torch
import scipy.signal

class ECGDataset(torch.utils.data.Dataset):
    """
    A torch dataset of ECG examples.

    Arguments:
        cfg: The config dict, like config.py's cfg["dataloader"].
        split: The name of the split (e.g., 'train', 'valid', 'test').
        output: The output directory where the processed files will be saved (optional).
        return_labels: Whether to return labels with waveforms. Useful to turn off
            during evaluation.
        all_waveforms: Whether to use all waveform files in a directory, instead of 
            using a specific label file.
        
    Calling Trainer.train() will train a full model; calling Trainer.run_eval_on_split()
    and Trainer.run_eval_on_all() evaluates the model on a list of files or a directory
    of files.
    """
    
    def __init__(
        self,
        cfg,  # Configuration dictionary passed when initializing the dataset
        split="train",  # Split name (e.g., 'train', 'valid', 'test')
        output=None,  # Output directory for saving processed files (optional)
        return_labels=True,  # Whether to return labels with the waveform data
        all_waveforms=False  # If True, load all waveforms in the directory, ignoring the label file
    ):
        self.cfg = cfg  # Store the passed config as an instance variable
        self.split = split  # The data split ('train', 'valid', 'test')
        self.output = output  # Output directory where files can be saved
        self.return_labels = return_labels and not all_waveforms  # Return labels unless using all waveforms without a label file
        self.all_waveforms = all_waveforms  # Whether to load all waveforms without filtering by a label file

        self.load_filelist()  # Load the list of files based on the config and split
        if self.return_labels:
            # If labels are being returned, load each label specified in 'label_keys'
            for label in self.cfg["label_keys"]:
                self.load_label(label)  # Call the load_label function for each label
        self.save_filelist()  # Optionally save the list of files
        self.get_mean_and_std()  # Calculate the mean and std of the waveforms for normalization
        self.filenames = self.filelist[self.cfg["filekey"]]  # Extract the filenames using the key specified in the config
        print(self.filelist, flush=True)  # Print the file list (flush=True ensures immediate printing)

    def __getitem__(self, index):
        """
        Returns the waveform and label (if applicable) at the specified index.
        """
        waveform = self.get_waveform(index)  # Get the waveform data using the index

        if self.return_labels:
            # Retrieve the label(s) corresponding to the waveform at the specified index
            y = self.filelist[self.cfg["label_keys"]].loc[[index]].to_numpy().flatten()

            # Apply normalization to the labels if specified in the config
            if self.cfg.get("normalize_y", False):  # If normalize_y is True, normalize the labels
                y = (y - np.mean(y)) / np.std(y)  # Normalize: (label - mean) / std

            y = torch.from_numpy(y).float()  # Convert labels to a PyTorch tensor (float)
            return [waveform, y]  # Return both waveform and label
        else:
            return [waveform]  # If labels are not returned, just return the waveform

    def __len__(self):
        """
        Returns the total number of files in the dataset.
        """
        return len(self.filelist)  # Return the length of the filelist (number of samples)

    def load_filelist(self):
        """
        Loads the list of files for the dataset based on the configuration and split.
        """
        if self.all_waveforms:
            # If using all waveforms, load all files in the waveform directory
            print("Running on all files", flush=True)
            self.filelist = pd.DataFrame(
                {
                    # Create a DataFrame where the column specified by 'filekey' contains all filenames
                    self.cfg["filekey"]: [s.split("/")[-1] for s in glob.glob(os.path.join(self.cfg["waveforms"], "*"))],
                    "split": "all"  # Mark all files as part of the 'all' split
                }
            )
        else:
            # Otherwise, load the file list from the label file specified in the config
            label_csv = self.cfg["label_file"]  # Get the label file path from the config
            print(f"Loading from {label_csv}", flush=True)  # Print which label file is being loaded
            self.filelist = pd.read_csv(label_csv)  # Load the label CSV file into a DataFrame

        print(f"{len(self.filelist)} files in list", flush=True)  # Print the total number of files

        # Filter files based on the split (e.g., 'train', 'valid', 'test')
        if self.split != "all":
            if self.cfg["crossval_idx"]:
                # If cross-validation is being used, adjust the split labels accordingly
                self.filelist["split"][self.filelist["split"] == "valid"] = "train"
                self.filelist["split"][self.filelist["split"] == f"train{self.cfg['crossval_idx']}"] = "valid"
            # Filter files to only include those in the current split
            self.filelist["split"] = self.filelist["split"].str.contains("train")
            self.filelist = self.filelist[self.filelist["split"] == self.split]

        print(f"{len(self.filelist)} files in split {self.split}", flush=True)  # Print the number of files in the current split

        # Apply label-based removal if specified in the config
        if self.cfg["remove_labels"] and self.return_labels:
            # Load the overread CSV that contains additional labels for exclusion
            overreads = pd.read_csv(self.cfg["overread_csv"])
            # Merge the overreads file with the filelist on the key specified in the config
            self.filelist = self.filelist.merge(overreads, how="left", on=self.cfg["filekey"], suffixes=("", "_y"))
            print(f"{len(self.filelist)} files after merging with overreads", flush=True)  # Print the number of files after merging

            # Remove any files that have labels to be excluded
            for remove_label in self.cfg["remove_labels"]:
                self.filelist = self.filelist[~self.filelist[remove_label].fillna(False)]  # Exclude based on the remove_label column

        print(f"{len(self.filelist)} files without removal criteria", flush=True)  # Print final count of files after exclusions

        self.filelist.reset_index(drop=True, inplace=True)  # Reset the index of the DataFrame for cleaner access

    def load_label(self, label):
        """
        Loads and processes the label(s) for the dataset.
        """
        for label in self.cfg["label_keys"]:
            if self.cfg["binary"]:
                if self.cfg["binary_positive_class"] == "below":
                    self.filelist[label] = self.filelist[label] <= self.cfg["binary_cutoff"]
                elif self.cfg["binary_positive_class"] == "above":
                    self.filelist[label] = self.filelist[label] >= self.cfg["binary_cutoff"]
                print(f"{self.filelist[label].sum()} positive examples in class {label}", flush=True)
            else:
                if self.cfg.get("normalize_y", False):  # Added .get to provide a default value of False
                    self.filelist[label] = (
                        self.filelist[label] - np.mean(self.filelist[label])
                    ) / np.std(self.filelist[label])
                print(f"Mean {self.filelist[label].mean()}, std {self.filelist[label].std()} in class {label}", flush=True)

    def save_filelist(self):
        """
        Saves the filelist to the output directory if it is provided.
        """
        if self.output:
            fname = os.path.join(self.output, "{}_filelist.csv".format(self.split))  # Filename based on the split
            if not os.path.exists(fname):
                self.filelist.to_csv(fname)  # Save the filelist if the file doesn't already exist

    def get_waveform(self, index):
        """
        Loads and processes the waveform file at the given index.
        """
        f = self.filelist[self.cfg["filekey"]][index]  # Get the filename at the specified index
        if "." not in f:
            f = f"{f}.{self.cfg['waveform_type']}"  # Append the correct file extension if not present
        f = os.path.join(self.cfg["waveforms"], f)  # Construct the full path to the waveform file
        x = np.load(f).astype(np.float32)  # Load the waveform as a numpy array of type float32

        if len(x) <= 12:
            x = x.T  # Transpose if there are 12 or fewer leads
        if len(x) > 5000:
            x = x[:5000]  # Crop the signal to a maximum length of 5000 samples

        # Apply filters and downsample based on the config
        if self.cfg["notch_filter"]:
            x = self.notch(x)  # Apply notch filter if specified
        if self.cfg["baseline_filter"]:
            x = self.baseline(x)  # Apply baseline filter if specified
        if self.cfg["downsample"]:
            x = x[::self.cfg["downsample"]]  # Downsample the signal if specified
            expected_length = 5000 // self.cfg["downsample"]
        else:
            expected_length = 5000

        # Check that the signal is of the correct length
        assert len(x) >= expected_length, (
            f"Got signal of length {len(x)}, which is too short for expected_length {expected_length}")
        assert (len(x) < 2 * expected_length) and not self.cfg["accept_all_lengths"], (
            f"Got signal of length {len(x)}, which looks too long for expected_length {expected_length}")

        x = x[:expected_length]  # Ensure the signal is the expected length

        # Normalize the signal if specified in the config
        if self.cfg["normalize_x"]:
            x = (x - self.cfg["x_mean"][:x.shape[1]]) / self.cfg["x_std"][:x.shape[1]]
        x[np.isnan(x)] = 0  # Replace NaNs with 0
        x[x == np.inf] = x[x != np.inf].max()  # Replace inf values with the max
        x[x == -np.inf] = x[x != -np.inf].min()  # Replace -inf values with the min

        if self.cfg["leads"]:
            x = x[:, self.cfg["leads"]]  # Select only the specified leads if provided
        x = x.T  # Transpose the final signal to match the expected format

        return torch.from_numpy(x).float()  # Convert the numpy array to a PyTorch tensor (float32)

    def notch(self, data):
        """
        Applies a notch filter to the waveform data.
        
        The notch filter is used to remove a narrow band of frequencies from the signal.
        In this case, it's a simple filter designed for basic processing.
        """
        data = data.T  # Transpose the data to work with each lead individually
        upsample = 5000 // data.shape[1]  # Upsampling factor based on input data shape
        sampling_frequency = 500  # Set the sampling frequency (500 Hz)
        row, __ = data.shape  # Get the number of leads (rows)
        processed_data = np.zeros(data.shape)  # Create an array to store the processed data
        b = np.ones(int(0.02 * sampling_frequency)) / 50.  # Design the moving average filter
        a = [1]  # No poles in the filter

        for lead in range(0, row):
            # If upsampling is needed, resample, filter, and then downsample the signal
            if upsample and upsample != 1:
                X = scipy.signal.resample(data[lead, :], 5000)  # Resample the signal to 5000 points
                X = scipy.signal.filtfilt(b, a, X)  # Apply the moving average filter
                X = X[::upsample]  # Downsample the signal back
            else:
                # Apply the filter directly if no upsampling is needed
                X = scipy.signal.filtfilt(b, a, data[lead, :])
            
            processed_data[lead, :] = X  # Store the filtered data

        return processed_data.T  # Transpose the data back to its original shape

    def baseline(self, data):
        """
        Applies a baseline wander removal filter to the waveform data using median filtering.
        
        The function estimates the baseline by applying median filters with different
        window sizes and subtracts the estimated baseline from the original signal.
        """
        data = data.T  # Transpose the data to process each lead independently
        row, __ = data.shape  # Get the number of leads (rows)
        sampling_frequency = data.shape[1] // 10  # Estimate the sampling frequency from the number of samples

        # First median filter to capture short-term variations (0.2 * sampling frequency window size)
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.ndimage.median_filter(data, [1, win_size], mode="constant")

        # Second median filter to capture long-term variations (0.6 * sampling frequency window size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.ndimage.median_filter(baseline, [1, win_size], mode="constant")

        # Subtract the estimated baseline from the original data to remove baseline wander
        filt_data = data - baseline

        return filt_data.T  # Transpose the filtered data back to its original shape

    def get_mean_and_std(self, batch_size=128, samples=8192):
        """
        Calculates and sets the mean and standard deviation of the ECG data for normalization.

        Arguments:
            batch_size: Number of samples per batch for the DataLoader.
            samples: Number of samples to use for calculating the mean and standard deviation.

        This method first checks if the mean and std are already computed or if normalization is 
        not required. If not, it selects a random subset of the dataset, loads it in batches, 
        and computes the mean and standard deviation for normalization purposes.
        """
        
        # If mean and standard deviation already exist in the config or normalization is not required, return
        if ("x_mean" in self.cfg and "x_std" in self.cfg) or not self.cfg["normalize_x"]:
            return
        
        # Create a deep copy of the config to avoid modifying the original
        cfg = copy.deepcopy(self.cfg)
        
        # Temporarily set `return_labels` and `normalize_x` to False while calculating statistics
        self.cfg["return_labels"], self.cfg["normalize_x"] =  False, False

        # Randomly select a subset of indices from the dataset for calculating statistics
        indices = np.random.choice(len(self), min(len(self), samples), replace=False)
        
        # Create a DataLoader to iterate over the sampled subset of data
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self, indices),  # Subset of dataset
            batch_size=batch_size,  # Batch size
            num_workers=self.cfg["n_dataloader_workers"],  # Number of parallel workers
            shuffle=False  # No need to shuffle as we're computing statistics
        )

        n = 0  # Total number of samples
        s1 = 0.  # Sum of the data
        s2 = 0.  # Sum of the squared data

        print("loading mean and std", flush=True)  # Print the status
        
        # Loop through the data in batches
        for x in tqdm.tqdm(dataloader):  # tqdm is used for progress bar display
            # Reshape and accumulate data statistics
            x = x[0].transpose(0, 1).reshape(x[0].shape[1], -1)  # Reshape to (channels, -1)
            n += np.float64(x.shape[1])  # Count total number of samples
            s1 += torch.sum(x, dim=1).numpy().astype(np.float64)  # Accumulate sum of data
            s2 += torch.sum(x ** 2, dim=1).numpy().astype(np.float64)  # Accumulate sum of squared data
        
        # Calculate the mean and standard deviation from the accumulated sums
        x_mean = (s1 / n).astype(np.float32)  # Mean of the data
        x_std = np.sqrt(s2 / n - x_mean ** 2).astype(np.float32)  # Standard deviation of the data

        # Print a warning if fewer samples than expected were used
        if n < samples:
            print(f"WARNING: calculating mean and std based on {n} waveforms", flush=True)

        # Output the calculated mean and std for verification
        print(f"Means: {x_mean}")
        print(f"Stds: {x_std}")

        # Restore the original configuration from the deep copy
        self.cfg = cfg
        
        # Store the calculated mean and std in the config for future normalization
        self.cfg["x_mean"] = x_mean
        self.cfg["x_std"] = x_std

