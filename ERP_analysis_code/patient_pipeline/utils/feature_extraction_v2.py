import numpy as np
import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Configurations original
# clf_ival_boundaries = np.arange(0.1, 0.51, 0.05)
# is_channel_prime = True
# ival_length = 50

# fe_info = {
#     'time_ivals': clf_ival_boundaries,
#     'time_val_length': ival_length,
#     'is_channel_prime': is_channel_prime
# }


# Configurations v2
clf_ival_boundaries_v2 = np.arange(0.1, 0.81, 0.05)
is_channel_prime = True
ival_length = 50

fe_info_v2 = {
    'time_ivals': clf_ival_boundaries_v2,
    'time_val_length': ival_length,
    'is_channel_prime': is_channel_prime
}

# Functions

# From BCI course
def get_jumping_means(epo, boundaries):
    """Feature extraction by averaging over time intervals between the given 'boundaries' """
    orig = epo.get_data()
    shape_orig = orig.shape
    X = np.zeros((shape_orig[0], shape_orig[1], len(boundaries)-1))
    for i in range(len(boundaries)-1):
        idx = epo.time_as_index((boundaries[i], boundaries[i+1]))
        idx_range = list(range(idx[0], idx[1]))
        X[:,:,i] = orig[:,:,idx_range].mean(axis=2)
    return X


def load_features_chached_v2(pickle_path 
                          #fe_info = dict()
                          ):
    """ New ival bounds: 0.1-0.81 in 50 ms steps
    """

    safe_name = _safe_filename(session_path=pickle_path) # replace \ and / by _
    print("Original file: ",safe_name)

    original_dir = os.path.dirname(pickle_path)  # Get directory 
    features_dir = os.path.join(original_dir, "features_v2")
    os.makedirs(features_dir, exist_ok=True)
    safe_name = _safe_filename(session_path=os.path.basename(pickle_path))

    cache_path = os.path.join(features_dir, "features_v2_" + safe_name)
    print("Corresponding .pkl file: ",cache_path)

    # check if a .pkl file for features already exists
    if os.path.exists(cache_path):
        print("A .pkl file already exists. Loading the data from {}".format(cache_path))
        with open(cache_path, 'rb') as f:
            features_info = pickle.load(f)

    # if not, then load the data and store it in a new .pkl file 
    else:
        print("A .pkl file does not exist yet. Loading the data and creating {}... (this might take a few mins)".format(cache_path))
        features_info = load_features_v2(pickle_path)  
        with open(cache_path, 'wb') as f:
            pickle.dump(features_info, f)
            
    return features_info

def load_features_v2(pickle_path):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data_info = pickle.load(f)
    else:
        print("Cannot read the pickle_path")
        raise FileNotFoundError(f"Cannot read the pickle path: {pickle_path}")

    trials = data_info.get("trials") 
    features = []
    ival_bounds = fe_info_v2.get('time_ivals')

    # Feature extraction
    for trial in trials:
        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                x1 = get_jumping_means(iteration[s],ival_bounds)
                # if is_channel_prime:
                x2 = x1.transpose(0,2,1) # make channel prime - to speed up I commented out the if statement
                x3 = x2.flatten()
                x4 = x3.reshape(1,-1)
                x = x4 
                features.append(x)

    # Store metadata
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    features = np.array(features)

    features_info = {
        "features": features,
        "fe_info": fe_info_v2,
        "pickle_path": pickle_path,
        "timestamp": timestamp
    }
    return features_info

## Functions to store data in pickl files
def _safe_filename(session_path):
    """replace / and \\ in data paths by underscores (to make a valid file name)"""
    return re.sub(r"[\\/]", "_", session_path)

def merge_features(feature_info_s1, feature_info_s2):

    print(feature_info_s1.keys())
    features_s1 = feature_info_s1.get('features')
    fe_info_s1 = feature_info_s1.get('fe_info')
    pickle_path_s1 = feature_info_s1.get('pickle_path')
    timestamp_s1 = feature_info_s1.get('timestamp')

    features_s2 = feature_info_s2.get('features')
    fe_info_s2 = feature_info_s2.get('fe_info')
    pickle_path_s2 = feature_info_s2.get('pickle_path')
    timestamp_s2 = feature_info_s2.get('timestamp') 

    # merge fe_info
    same = True
    time_info_keys = ['time_ivals', 'time_val_length', 'is_channel_prime']
    for key in time_info_keys:
        if key == 'time_ivals':
            if not np.array_equal(fe_info_s1.get(key), fe_info_s2.get(key)):
                same = False

        elif fe_info_s1.get(key) != fe_info_s2.get(key):
            same = False
    if same:        
        fe_info = fe_info_s1
    else:
        print("Caution! The data files of both sessions do not have the same feature extraction settings!")
        fe_info = [fe_info_s1, fe_info_s2]

    # store result
    feature_info = {
    "features": np.concatenate((features_s1, features_s2), axis=0),
    "fe_info": fe_info,
    "pickle_path": [pickle_path_s1, pickle_path_s2],
    "timestamp": [timestamp_s1, timestamp_s2]
    }

    return feature_info

def load_or_extract_markers(pickle_path, online_trials):
    #print("Original file: ",pickle_path)
    original_dir = os.path.dirname(pickle_path)  # Get directory 
    markers_dir = os.path.join(original_dir, "online")
    os.makedirs(markers_dir, exist_ok=True)
    safe_name = _safe_filename(session_path=os.path.basename(pickle_path))

    cache_path = os.path.join(markers_dir, "markers_v1_" + safe_name)
    #print("Corresponding .pkl file: ",cache_path)

    # check if a .pkl file for features already exists
    if os.path.exists(cache_path):
        #print("A .pkl file already exists. Loading the data from {}".format(cache_path))
        with open(cache_path, 'rb') as f:
            markers_info = pickle.load(f)

    # if not, then load the data and store it in a new .pkl file 
    else:
        #print("A .pkl file does not exist yet. Loading the data and creating {}... (this might take a few mins)".format(cache_path))
        markers_info = load_markers(online_trials)  
        with open(cache_path, 'wb') as f:
            pickle.dump(markers_info, f)
            
    return markers_info

def load_markers(online_trials):

    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]       
    markers2 = np.zeros(len(online_labels), np.int8)
    epoch_c = 0
    for trial in online_trials:
        for iteration in trial:
            for e, epoch in enumerate(iteration):
                markers2[epoch_c] = iteration[e].events[0,2]
                #print(iteration[e].events[0,2])
                epoch_c+=1

    markers_info={
        "markers": markers2
    }
    return markers_info

def epoch_vectorizer_channelprime(raw_calibration_trials, ival_bounds):
    """
    Extracts features from raw_calibration_trials and returns (channel prime) X and (labels) y to feed into classifier
    
    Parameters
    - raw_calibration_trials (list): nested list of trials. Each trial is a list of (15 or less) iterations. Each iteration is a list of 6 epochs
    - ival_bounds (np array): time interval boundaries between which the time points are averaged.

    Output:
    - X (np array): 2D numpy array of shape (epochs, features) where 
    - y (np array): 1D numpy array of labels of each epoch, e.g. [0 0 0 0 1 0 1 0]
    """
    # Average time points over the time intervals in the given interval boundaries (ival_bounds)
    calibration_trials = [[get_jumping_means(iteration, ival_bounds) for iteration in trial] for trial in raw_calibration_trials]
    calibration_trials_reshaped = [
        [epochs.transpose(0, 2, 1) for epochs in trial] # make epochs channel prime
        for trial in calibration_trials
    ]   
    # Flatten to get X of shape (samples, features)
    # Flatten each epoch from 2D (n_time_ivals, n_channels) to 1D (n_time_ivals * n_channels) 
    calibration_stimuli = [epoch.reshape(-1) for trial in calibration_trials_reshaped for iteration in trial for epoch in iteration]
    calibration_stimuli = np.array(calibration_stimuli) # shape (samples,features),  e.g. (1038, 124)

    # Obtain labels for each epoch
    calibration_labels = [(1 if event > 107 else 0) for trial in raw_calibration_trials for iteration in trial for event in iteration.events[:,2]] 
    calibration_labels = np.array(calibration_labels) # contains only 0's and 1's. # shape (samples,) e.g. (1038,)

    X = calibration_stimuli
    y = calibration_labels
    return X, y