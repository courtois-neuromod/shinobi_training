import os.path as op
import os
import glob
import argparse
from bids_loader.stimuli.game import get_variables_from_replay
import retro
import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime
from shinobi_behav.features.features import fix_position_resets
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--datapath",
    default='.',
    type=str,
    help="Data path to look for .bk2 files. Should be the root of the shinobi dataset.",
)

def create_info_dict(repvars, dataset_info_path):
    """
    Creates a dictionary containing information about a replay.

    Parameters:
        repvars (dict): A dictionary containing replay variables.
        dataset_info_path (str): The path to the dataset info JSON file.

    Returns:
        dict: A dictionary containing information about the replay, including the subject ID, session ID, repetition level,
        whether the level was cleared, the duration of the repetition, the final score, the amount of health lost, the percent
        complete, and whether the repetition was fake or not.
    """
    # Init dict
    info_dict = {}

    # Add subject ID
    replayfile = repvars["filename"]
    info_dict["SubjectID"] = replayfile.split("/")[-1].split("_")[0]

    # Add session ID
    info_dict["SessionID"] = replayfile.split("/")[-1].split("_")[1]

    # Add repetition level
    info_dict["Level"] = replayfile.split("/")[-1].split("_")[3]

    # Add cleared
    lives_lost = sum([x for x in np.diff(repvars["lives"], n=1) if x < 0])
    if lives_lost == 0:
        cleared = True
    else:
        cleared = False
    info_dict["Cleared"] = cleared

    # Add repetition duration
    info_dict["Duration"] = len(repvars["X_player"])/60

    # Add final score (performance)
    info_dict["FinalScore"] = repvars["score"][-1]

    # Add amount of health lost (performance)
    diff_health = np.diff(repvars["health"], n=1)
    try:
        index_health_loss = list(np.unique(diff_health, return_counts=True)[0]).index(-1)
        total_health_loss = np.unique(diff_health, return_counts=True)[1][index_health_loss]
    except Exception as e:
        print(e)
        total_health_loss = 0
    info_dict["TotalHealthLost"] = total_health_loss

    # Add percent complete
    max_pos = load_max_pos(info_dict['Level'], dataset_info_path)
    end_of_level = max_pos - 300 # We assume the level is complete is the player reaches max_pos - 300
    x_pos_clean = fix_position_resets([repvars])[0]
    percent_complete = x_pos_clean[-1]/end_of_level*100
    info_dict["PercentComplete"] = percent_complete

    # Add if fakerep
    if info_dict["FinalScore"] < 200:
        fakerep = True
    else:
        fakerep = False

    info_dict["FakeRep"] = fakerep

    return info_dict

def add_info_to_json(json_dict, repvars, dataset_info_path):
    """
    Adds information to a JSON dictionary and returns it.

    Parameters:
        json_dict (dict): The JSON dictionary to be updated.
        repvars (dict): A dictionary with the variables of one replay.
        dataset_info_path (str): The path to the dataset information file.

    Returns:
        dict: The updated JSON dictionary.
    """
    # create json sidecar
    info_dict = create_info_dict(repvars, dataset_info_path)

    # Add number of days of training (uses info from the original json)
    subject = repvars['filename'].split("/")[-1].split("_")[0]
    first_day = datetime.fromtimestamp(load_first_day_of_training(subject, dataset_info_path))
    current_day = datetime.fromtimestamp(json_dict["LevelStartTimestamp"])
    n_days_of_training = current_day - first_day
    json_dict["DaysOfTraining"] = n_days_of_training.days + 1
    
    # Populate json with new keys
    for key in info_dict.keys():
        json_dict[key] = info_dict[key]
    return json_dict

def load_max_pos(level, dataset_info_path):
    """
    Load the maximum X position for a given level from the dataset_info.json file.

    Args:
        level (int): The level for which to load the maximum X position.
        dataset_info_path (str): The path to the dataset info JSON file.

    Returns:
        float: The maximum X position for the given level.
    """
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    max_pos = dataset_info["Maximum X position"][level]
    return max_pos

def load_first_day_of_training(subject, dataset_info_path):
    """
    Load the timestamp of the first day of training for a given subject.

    Args:
        subject (str): The subject ID.
        dataset_info_path (str): The path to the dataset info JSON file.

    Returns:
        str: The timestamp of the first day of training for the given subject.
    """
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    first_day_timestamp = dataset_info["Training start timestamp"][subject]
    return first_day_timestamp


if __name__ == "__main__":
    args = parser.parse_args()
    DATA_PATH = args.datapath
    dataset_info_path = op.join(DATA_PATH, "code", "dataset_info.json")

    replayfile_list = sorted(glob.glob(op.join(DATA_PATH, "*", "*", "*", "*.bk2")))
    for replayfile in replayfile_list:
        try:
            # Load json sidecar
            with open(replayfile.replace(".bk2", ".json"), "r") as infile:
                json_dict = json.load(infile)
            if "Cleared" not in json_dict.keys(): # Check if some key is in the dict
                print('Processing {}'.format(replayfile))
                repvars = get_variables_from_replay(replayfile, skip_first_step=True, save_gif=True, game="ShinobiIIIReturnOfTheNinjaMaster-Genesis", inttype=retro.data.Integrations.STABLE)
                json_dict = add_info_to_json(json_dict, repvars, dataset_info_path)
                with open(replayfile.replace(".bk2", ".json"), "w") as outfile:
                    json.dump(json_dict, outfile, default=str)
        except Exception as e:
            print(e)