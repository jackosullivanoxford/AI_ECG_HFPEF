"""Functions and default values for configuration."""
import collections

def update_config_dict(d, u):
    """Updates a nested dict d based on a second nested dict u."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_config_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dict_to_str(d, prefix=""):
    """Converts a dict to a sanitized string which can be included in filenames."""
    if isinstance(d, collections.Mapping):
        return ",".join(dict_to_str(d[k], prefix=f"{prefix}{k}_") for k in sorted(d))
    else:
        out = str(d).replace("[", "-").replace("]", "-").replace("/", "_")
        for c in '\\:*?"<>|$':
            out = out.replace(c, "")
        return f"{prefix}{out}"


cfg = {
    "optimizer": {
        "optimizer": "adam",
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "n_epochs": 1000,
        # Whether to use ReduceLROnPlateau
        "reduce_on_plateau": True,
        # Patience for ReduceLROnPlateau
        "patience": 5,
        # Max reduction in LR by ReduceLROnPlateau
        "max_reduction": 1e-2,
        # If not `reduce_on_plateau`, number of steps for StepLR
        "lr_plateaus": 4,
        "save_all_checkpoints": False,
        # Path to save models and logs to
        "save_path":
            "/oak/stanford/groups/euan/projects/jackos/ecg/files_from_git_AI_ECG_HFPEF/output/",
    },

    "dataloader": {
        # Whether the task is binary
        "binary": True,
        # Cutoff to binarize a continuous label at (.5 if label is already binary)
        "binary_cutoff": .5,
        # Whether the positive class is above or below the cutoff {"above", "below"}
        "binary_positive_class": "above",
        # A list of columns in `overread_csv` to use as exclusion criteria
        # (Used whenever a label_file is used)
        "remove_labels": [],
        # Whether to compute means and stds to lead-wise normalize
        "normalize_x": True,
        # Whether to normalize a continuous y value
        "normalize_y": False,
        # Whether to notch filter the waveforms
        "notch_filter": False,
        # Whether to filter the waveforms for baseline wander
        "baseline_filter": False, 
        # Ammount to downsample the wavefrom signal
        "downsample": 2,
        # Whether to ignore errors with signal length
        "accept_all_lengths": False,
        # Either None to use all 12 leads, or a list of lead indicies e.g. `[0]` to use
        # a subset of the leads
        "leads": None,  
        "n_dataloader_workers": 32,
        # If using cross-validation, the cross-validation index
        "crossval_idx": None,
        # A CSV of overreads, for use with `remove_labels`.
        "overread_csv": "/path/to/overreads",
        # The directory containing the waveforms
        "waveforms": "/oak/stanford/groups/euan/projects/ECGs/ECGs/processed",
        # The waveform filetype.
        "waveform_type": "npy",
        # The column name containing filenames in `label_file`
        "filekey": "deid_filename",
    },

    "model": {
        # The name of the model architecture (see `model_specs.py`)
        "model_type": "seer",
        # Whether the model_spec is 1d or 2d
        "is_2d": True,
        "conv_width": 3,
        "drop_prob": 0.,
        "batch_norm": True,
        # The weight for positive examples
        "pos_weight": 1.,
    },
}

# Examples of task configs, which will overwrite the above config. 
task_cfg = {
    "ICDHFpEF": {
        "dataloader": {
            # Since the labels here are already binary, no need to overwrite binarization
            "task": "ICDHFpEF",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_ICDHFpEF.csv",
            "label_keys": ["ICDHFpEF"],
        }
    },
    "early_HFpEF": {
        "dataloader": {
            # Since the labels here are already binary, no need to overwrite binarization
            "task": "early_HFpEF",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_early_HFpEF.csv",
            "label_keys": ["early_HFpEF"],
        }
    },
    "indeter_HFpEF": {
        "dataloader": {
            "task": "indeter_HFpEF",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_indeter_HFpEF.csv",
            "label_keys": ["indeter_HFpEF"],
        }
    },
    "late_HFpEF": {
        "dataloader": {
            "task": "late_HFpEF",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_late_HFpEF.csv",
            "label_keys": ["late_HFpEF"],
        }
    },
    "ICDDM": {
        "dataloader": {
            "task": "ICDDM",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_ICDDM.csv",
            "label_keys": ["ICDDM"],
        }
    },
    "ICDHFNOS": {
        "dataloader": {
            "task": "ICDHFNOS",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_ICDHFNOS.csv",
            "label_keys": ["ICDHFNOS"],
        }
    },
    "subclinical_HFpEF": {
        "dataloader": {
            "task": "subclinical_HFpEF",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_subclinical_HFpEF.csv",
            "label_keys": ["subclinical_HFpEF"],
        }
    },
    "ICDHTN": {
        "dataloader": {
            "task": "ICDHTN",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_ICDHTN.csv",
            "label_keys": ["ICDHTN"],
        }
    },
    "subclinical_HFpEF_strict": {
        "dataloader": {
            "task": "subclinical_HFpEF_strict",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_subclinical_HFpEF_strict.csv",
            "label_keys": ["subclinical_HFpEF_strict"],
        }
    },
    "H2PEF_imp_excludes": {
        "dataloader": {
            "task": "H2PEF_imp_excludes",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_H2PEF_imp_excludes.csv",
            "label_keys": ["H2PEF_imp_excludes"],
            "normalize_y": True,  # Optional, depending on the data
        }
    },
    "PASP_imp1": {
        "dataloader": {
            "task": "PASP_imp1",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_PASP_imp1.csv",
            "label_keys": ["PASP_imp1"],
            "normalize_y": True,  # Optional, depending on the data
        }
    },
    "E_e_used_imp": {
        "dataloader": {
            "task": "E_e_used_imp",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_E_e_used_imp.csv",
            "label_keys": ["E_e_used_imp"],
            "normalize_y": True,  # Optional, depending on the data
        }
    },
    "H2PEF_imp": {
        "dataloader": {
            "task": "H2PEF_imp",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_H2PEF_imp.csv",
            "label_keys": ["H2PEF_imp"],
            "normalize_y": True,  # Optional, depending on the data
        }
    },
    # Adding missing binary targets
    "late_HFpEF_ICD": {
        "dataloader": {
            "task": "late_HFpEF_ICD",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_late_HFpEF_ICD.csv",
            "label_keys": ["late_HFpEF_ICD"],
        }
    },
    "ICD_HFNOS_diastolic": {
        "dataloader": {
            "task": "ICD_HFNOS_diastolic",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_ICD_HFNOS_diastolic.csv",
            "label_keys": ["ICD_HFNOS_diastolic"],
        }
    },
    "ICDHFNOS": {
        "dataloader": {
            "task": "ICDHFNOS",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_ICDHFNOS.csv",
            "label_keys": ["ICDHFNOS"],
        }
    },
    "early_HFpEF_ICD": {
        "dataloader": {
            "task": "early_HFpEF_ICD",
            "label_file": "/oak/stanford/groups/euan/projects/jackos/ecg/targets_for_git_AI_ECG_HFPEF/target_early_HFpEF_ICD.csv",
            "label_keys": ["early_HFpEF_ICD"],
        }
    }
}

