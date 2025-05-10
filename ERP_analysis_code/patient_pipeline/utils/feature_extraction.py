import numpy as np

# See cell block in dump file for documentation on how this function works
def get_jumping_means(epo, boundaries):
    """Feature extraction by averaging over time intervals between the given 'boundaries' """
    shape_orig = epo.get_data().shape
    X = np.zeros((shape_orig[0], shape_orig[1], len(boundaries)-1))
    for i in range(len(boundaries)-1):
        idx = epo.time_as_index((boundaries[i], boundaries[i+1]))
        idx_range = list(range(idx[0], idx[1]))
        X[:,:,i] = epo.get_data()[:,:,idx_range].mean(axis=2)
    return X


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