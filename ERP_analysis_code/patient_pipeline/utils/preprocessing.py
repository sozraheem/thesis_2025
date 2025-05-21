import mne
import os
import re
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Configurations -----------------------------------------------------------------------
filter_band = (0.5, 16)
non_eeg_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
decimate = 10 # for downsampling
montage = "standard_1020"
filter_method = "iir"
only_eeg = True
tmin = -0.2 # Default of mne.Epochs
tmax = 1
baseline = None
channels_to_discard = ["AF7","AF3","AF4","AF8","F5","F1","F2","F6","FT7","FC3","FCz","FC4","FT8","C5","C1","C2","C6","TP7","CP3","CPz","CP4","TP8","P5","P1","P2","P6","PO7","PO5","POz","PO6","PO8","Oz"] # same as Ch33 - Ch64 (the channels that are not found in the online files)

# Functions ----------------------------------------------------------------------------

# Obtained from assignment 7 of the BCI course
def _load_and_preprocess_raw(header_file, filter_band=filter_band):
    raw = mne.io.read_raw_brainvision(header_file, misc=non_eeg_channels, preload=True)
    raw.set_montage(montage)
    raw.filter(*filter_band, method=filter_method)
    raw.pick_types(eeg=only_eeg)
    return raw

# Obtained from assignment 7 of the BCI course
def _epoch_raw(raw, decimate=decimate):
    target_ids = list(range(111, 117))     # [111, 112, 113, 114, 115, 116]
    non_target_ids = list(range(101, 107)) # [101, 102, 103, 104, 105, 106]
    event_id = {f"Word_{i-110}/Target": i for i in target_ids} # {'Word_1/Target': 111, ..., 'Word_6/Target': 116}
    event_id.update({f"Word_{i-100}/NonTarget": i for i in non_target_ids})
    
    evs = mne.events_from_annotations(raw)[0]
    # print(evs.shape) # e.g. (548,3)
    # print(evs)
    # e.g.
    # [[    0      0  99999]
    # [  4688      0    203]
    # [  9436      0    103]
    # ...
    # [267206      0    106]
    # [267559      0    103]
    # [270989      0    255]]

    epoch = mne.Epochs(raw, events=evs, event_id=event_id, decim=decimate,
                       proj=False, tmax=tmax, baseline=baseline)
    return epoch

def all_have_same_condition(data_path, show_conditions = False, selection = None):
    """ 
    Checks if all runs within a session have the same condition (6D vs HP, 350 vs 250)
    
    Input:
    - data_path: path to the folder where .vhdr files are stored. E.g. "data_p1/P1_S3/anonymized"
    - show_conditions: if True, then print the condition per .vhdr file, i.e., per run
    - selection: load only the .vhdr runs with a certain condition. E.g. "6D_long_350"

    Output:
    - Boolean: True if all conditions are the same, False otherwise
    """

    # data_path = "data_p1/P1_S3/anonymized" 
    data_dir = Path.cwd() / data_path
    if selection is None:
        header_files = data_dir.glob("auditoryAphasia*.vhdr") 
    else:
        header_files = data_dir.glob("auditoryAphasia_" + selection + "*.vhdr") 
        
    header_files_list = list(header_files) # convert generator object into list
    header_files_names = [header.name for header in header_files_list] # obtain all headers as strings

    # parse condition of all header files (i.e., all runs)
    header_conditions_1 = [header.split("_")[1] for header in header_files_names] # first condition: 6D or HP
    header_conditions_2 = [header.split("_")[2] for header in header_files_names] # second condition: 350 or 250

    if show_conditions:
        print(header_conditions_1)
        print(header_conditions_2)

    return len(set(header_conditions_1))==1 and len(set(header_conditions_2))==1

def _list_iterations(raw_data):
    """
    For a single run containing 6 trials, compute for each trial the nr of iterations used (this is not always 15!)
    Returns a list of 6 elements (each elelement = n_iterations for that trial), e.g. [15 15 8 15 14 15]
    """

    evs = mne.events_from_annotations(raw_data)[0] 
    markers = evs[:,2] # list all event_ids

    # obtain all unique event markers
    uniques, first_indices, counts = np.unique([x for x in markers], return_index=True, return_counts=True)

    # Only keep trial start markers [200, 201, 202, 203, 204, 205] and the run end marker 255
    only_starter_markers = [x for x in uniques if x>199 and x<206 or x==255]
    only_starter_markers_indices = [first_indices[np.where(uniques==x)][0] for x in uniques if x>199 and x<206 or x==255]

    # Sort the markers according to their stimulus onset order in the trial
    sorted_starter_markers = [x for _,x in sorted(zip(only_starter_markers_indices,only_starter_markers))] 
    sorted_starter_markers_indices = np.sort(only_starter_markers_indices)

    stimuli_per_trial = np.diff(sorted_starter_markers_indices)-1
    iterations_per_trial = stimuli_per_trial/6

    return iterations_per_trial

def load_complete_session(data_path, selection = None, discard_channels = None):
    """Load and preprocess data of a complete session; return trials, iterations, epochs"""

    # data_dir = Path.cwd() / "data_p1/P1_S1/anonymized" 
    data_dir = Path.cwd() / data_path
    if selection is None:
        header_files = data_dir.glob("auditoryAphasia*.vhdr") # assuming the condition within a session is the same
        print("All conditions of this session are the same: ", all_have_same_condition(data_path)) # either True or False
    else:
        assert isinstance(selection, str), "given selection parameter is not an instance of String" 
        header_files = data_dir.glob("auditoryAphasia_" + selection + "*.vhdr")
        print("All conditions of the selected runs of this session: ",all_have_same_condition(data_path, show_conditions=True, selection=selection))

    # Load the data, preprocess and slice it into epochs
    epochs = list()

    # added: per run, store how many iterations were used for every trial in that run
    all_trial_iterations = list() 
    print("Number of iterations per trial:")
    run_count = 1 # added to keep track of runs

    filenames = list() # store metadata

    for f in header_files:
        raw_data = _load_and_preprocess_raw(f)

        # To store metadata in the pickle file
        parent1 = f.parent.name            # immediate parent ("anonymized")
        parent2 = f.parent.parent.name     # grandparent (e.g. "P1_S5")
        filenames.append(f"{parent2}/{parent1}/{f.name}")
        
        # discard channels (eventually move this feature into load_and_preprocess_raw?)
        if discard_channels:
            raw_data = raw_data.drop_channels(channels_to_discard)

        epochs.append(_epoch_raw(raw_data))

        # added
        iterations_per_trial = _list_iterations(raw_data) # list with nr of iterations per trial for all six trials
        print(f"Run {run_count}: {iterations_per_trial}")
        all_trial_iterations.append(iterations_per_trial.astype(int)) # store this per-run iteration counter list 
        run_count+=1

    print(f"Loaded files: {filenames}")

    # Overwrite epochs list to save memory
    epochs = mne.concatenate_epochs(epochs) 

    # Combine 6 epochs into a single iteration (6 stimuli together form a single iteration)
    iterations = [epochs[i:i+6] for i in np.arange(0, epochs.events.shape[0],6)] # for loop goes from 0 to final epoch in steps of 6
    # Assert that each iteration contains exactly 1 Target
    assert all([len(iteration["Target"]) == 1 for iteration in iterations]), "Number of targets in single iterations is unequal to 1."

    # Group the correct amount of iterations per trial (this is not always 15!)
    trials = []
    all_trial_iterations = np.concatenate(all_trial_iterations) # flatten all per-run iteration counter lists to a single 1D array
    idx = 0
    for n_iters in all_trial_iterations:
        trials.append(iterations[idx : idx + n_iters])
        idx += n_iters


    # Store metadata
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data_info = {
    "trials": trials,
    "iterations": iterations,
    "epochs": epochs,
    "filenames": filenames,
    "preprocessing": {
        "filter_band": filter_band,
        "non_eeg_channels": non_eeg_channels,
        "decimate": decimate,
        "montage": montage,
        "filter_method": filter_method,
        "only_eeg": only_eeg,
        "tmin": tmin,
        "tmax": tmax,
        "baseline": baseline,
        "channels_to_discard": channels_to_discard,
        "channels": epochs[0].info['ch_names'],
        "fs": epochs[0].info['sfreq'],
        # may be extended/updated later
        }, 
    "timestamp": timestamp
    }

    return data_info

def inspect_session(data_path):
    """Inspect data of a complete session, print relevant information"""

    # data_dir = Path.cwd() / "data_p1/P1_S1/anonymized" 
    data_dir = Path.cwd() / data_path
    header_files = data_dir.glob("auditoryAphasia*.vhdr") # assuming the condition within a session is the same
    print("Condition per run: ")
    print("All conditions of this session are the same: ", all_have_same_condition(data_path, show_conditions=True)) # print conditions

    # Per run, print how many iterations were used for every trial in that run
    print("Number of iterations per trial:")
    run_count = 0 # added to keep track of runs

    for f in header_files:
        raw_data = _load_and_preprocess_raw(f)
        iterations_per_trial = _list_iterations(raw_data) # obtain list with nr of iterations per trial for all six trials;
        print("Run {}: {}".format(run_count, iterations_per_trial.astype(int)))
        run_count+=1

def get_n_epochs(trials):
    """"Returns the total number of epochs found in the given trials"""
    n_epochs = 0
    for trial in trials:
        for iteration in trial:
            n_epochs += 6
    return n_epochs

def get_iteration_structure(trials):
    """Returns the nr of iterations for each trial in the given trials"""
    n_iterations = list()
    for trial in trials:
        n_iterations.append(len(trial)) # obtain trial length per trial
    n_iterations = np.array([n_iterations[i:i+6] for i in np.arange(0, len(trials),6)]) # group 6 trials into a run
    return n_iterations

def get_n_iterations(trials):
    """"Returns the total amount of iterations found in the given trials"""
    return np.sum(get_iteration_structure(trials))

def get_preprocessing(data):
    """
    Returns all preprocessing settings of the data stored in a list
    
    Parameters:
    - data (dictionary): data to extract preprocessing from (data is obtained as output of load_session_chached())

    Output:
    - list (list): all preprocessing settings in a list of strings. Each string is formatted as "{key}: {value}"
    """
    
    list = []
    for key in data.get('preprocessing').keys():
        list.append(f"{key}: {data.get('preprocessing').get(key)}")
    return list    

# Note: when changing something in the loading/preprocessing, the stored pickl files should be removed as they are outdated.
# to do: figure out if the note above can be done automatically 
def load_session_chached(session_path, cache_dir="cache/", selection = None, discard_channels = False):
    """Load preprocessed trials, iterations, and epochs either for the first time, saving it as a .pkl or if one exists, directly load from a .pkl file
    
    For visualization purposes, the `iterations` and `epochs` is also returned. However, this takes substantially more memory when storing, so these two features/returns might be removed later

    Input:
    - session_path: path to the session folder where .vhdr files are stored. E.g. "data_p1/P1_S3/anonymized"
    - selection: load only the .vhdr runs with a certain condition. E.g. "6D_long_350"
    - discard_channels: ONLY for the offline sessions! if True, then discard the extra channels of the offline sessions. if False, keep all channels. ('extra' channels = channels that are not found in the online sessions)

    Output: 
    - A single dictionary with the keys: (['trials', 'iterations', 'epochs', 'filenames', 'preprocessing', 'timestamp'])
        - trials: nested list of trials. Each trial is a list of (15 or less) iterations. Each iteration is a list of 6 epochs. 
        - iterations: nested list of iterations. Each iteration is a list of 6 epochs. 
        - epochs: list of all epochs concatenated from the loaded session.
        - filenames: "parent2/parent1/filename" for all loaded files
        - preprocessing: a dictionary with preprocessing configurations
        - timestamp: date and time when the pickle file has been created

    Example usage:
    # Example 1: Loading data from session 3
    > datapath = "data_p1/P1_S3/anonymized"
    > data = load_session_chached("data_p1/P1_S3/anonymized")
    > trials = data.get('trials')

    # Example 2: Loading data from sessions 1 and 2, only with the conditions 6D and 350. Also discard the extra channels (63 EEG --> 31 EEG).
    > data_s1 = load_session_chached("data_p1/P1_S1/anonymized", selection = "6D_long_350", discard_channels=True)
    > data_s2 = load_session_chached("data_p1/P1_S2/anonymized", selection = "6D_long_350", discard_channels=True)
    > data_s12 = merge_sessions(data_s1, data_s2)
    > calibration_trials = data_s12.get('trials')
    """

    os.makedirs(cache_dir, exist_ok=True)
    safe_name = _safe_filename(session_path=session_path) # replace \ and / by _
    print("Loading file: ",safe_name)
    if selection is None:
        if not discard_channels:
            cache_path = os.path.join(cache_dir, safe_name + ".pkl")
        else:
            cache_path = os.path.join(cache_dir, safe_name + "_dc" + ".pkl")    
    else:    
        assert isinstance(selection, str), "selection is not an instance of String"
        if not discard_channels:
            cache_path = os.path.join(cache_dir, safe_name + selection + ".pkl")
        else:
            cache_path = os.path.join(cache_dir, safe_name + selection + "_dc" + ".pkl")    

    print("Corresponding .pkl file: ",cache_path)

    # check if a .pkl file for that session already exists
    if os.path.exists(cache_path):
        print("A .pkl file already exists. Loading the data from {}".format(cache_path))
        with open(cache_path, 'rb') as f:
            data_info = pickle.load(f)

    # if not, then load the data and store it in a new .pkl file 
    else:
        print("A .pkl file does not exist yet. Loading the data and creating {}... (this might take a few mins)".format(cache_path))
        data_info = load_complete_session(session_path, selection, discard_channels)  
        with open(cache_path, 'wb') as f:
            pickle.dump(data_info, f)
            
    return data_info

## Functions to store data in pickl files
def _safe_filename(session_path):
    """replace / and \\ in data paths by underscores (to make a valid file name)"""
    return re.sub(r"[\\/]", "_", session_path)

# TO DO: test this function
def merge_sessions(data_s1, data_s2): 
    trials_s1 = data_s1.get('trials')
    iterations_s1 = data_s1.get('iterations')
    epochs_s1 = data_s1.get('epochs')
    filenames_s1 = data_s1.get('filenames')
    preprocessing_s1 = data_s1.get('preprocessing')
    timestamp_s1 = data_s1.get('timestamp')

    trials_s2 = data_s2.get('trials')
    iterations_s2 = data_s2.get('iterations')
    epochs_s2 = data_s2.get('epochs') 
    filenames_s2 = data_s2.get('filenames')
    preprocessing_s2 = data_s2.get('preprocessing')
    timestamp_s2 = data_s2.get('timestamp')
  
    if _have_same_preprocessing(preprocessing_s1, preprocessing_s2):
        preprocessing = preprocessing_s1
    else:
        print("Caution! The data files of both sessions do not have the same preprocessing settings!")
        preprocessing = [preprocessing_s1, preprocessing_s2]

    data_info = {
    "trials": trials_s1+trials_s2,
    "iterations": iterations_s1+iterations_s2,
    "epochs": mne.concatenate_epochs([epochs_s1,epochs_s2]),
    "filenames": filenames_s1+filenames_s2,
    "preprocessing": preprocessing,
    "timestamp": [timestamp_s1, timestamp_s2]
    }

    return data_info

# TO DO: test if this function works
def _have_same_preprocessing(pp_s1, pp_s2):
    return pp_s1 == pp_s2


