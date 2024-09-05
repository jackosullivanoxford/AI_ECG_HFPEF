class ECGDataset(torch.utils.data.Dataset):
    """
    A torch dataset of ECG examples.

    Arguments:
        cfg: The config dict, like config.py's cfg["dataloader"].
        split: The name of the split.
        output: The output directory.
        return_labels: Whether to return labels with waveforms. Useful to turn off
            during evaluation.
        all_waveforms: Whether to use all files in a directory, rather than using a label file.

    Calling Trainer.train() will train a full model; calling Trainer.run_eval_on_split()
    and Trainer.run_eval_on_all() evaluates the model on a list of files and a directory
    of files.
    """
    def __init__(
        self,
        cfg,
        split="train",
        output=None,
        return_labels=True,
        all_waveforms=False):

        self.cfg = cfg
        self.split = split
        self.output = output
        self.return_labels = return_labels and not all_waveforms
        self.all_waveforms = all_waveforms

        self.load_filelist()
        if self.return_labels:
            for label in cfg["label_keys"]:
                self.load_label(label)
        self.save_filelist()
        self.get_mean_and_std()
        self.filenames = self.filelist[cfg["filekey"]]
        print(self.filelist, flush=True)

    def __getitem__(self, index):
        waveform = self.get_waveform(index)

        if self.return_labels:
            y = self.filelist[self.cfg["label_keys"]].loc[[index]].to_numpy().flatten()

            # Apply normalization if specified in the config
            if self.cfg.get("normalize_y", False):
                y = (y - np.mean(y)) / np.std(y)

            y = torch.from_numpy(y).float()
            return [waveform, y]
        else:
            return [waveform]

