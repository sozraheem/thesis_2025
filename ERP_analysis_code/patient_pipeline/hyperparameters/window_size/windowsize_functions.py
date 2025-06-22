# Functions to pilot on the optimal moving window size
import numpy as np
from utils.preprocessing import load_session_chached, merge_sessions
from utils.feature_extraction_v2 import load_features_chached_v2, merge_features
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from toeplitzlda.classification import ToeplitzLDA


def merge_all_data_of_patient(patient,last_online, calibration_selection):
    """Merge data of all sessions"""

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"
        
    # start first with merging calibration sessions
    data_s1 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized", selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized", selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)

    calibration_selection_dc = f"{calibration_selection}_dc"
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)


    for session in range(3,last_online+1):
        data_new_session = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{session}/anonymized")
        data_train = merge_sessions(data_train, data_new_session)
        features_new_session = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{session}_anonymized.pkl")
        features_train = merge_features(features_train, features_new_session)

    return data_train, features_train

def print_metadata_pilot(all_data, all_features):
    print(f"Metadata of pilot patient:")
    print(f"-------------------- Dataset --------------------")
    print(f"Total number of trials: {len(all_data.get('trials'))}")
    print(f"Total number of iterations: {len(all_data.get('iterations'))}")
    print(f"Total number of epochs: {len(all_data.get('epochs'))}")
    # print("Loaded files:")
    # for f in all_data.get('filenames'):
    #     print(f)
    print("-------------------- Preprocessing --------------------")
    for p in (all_data.get('preprocessing').keys()):
        print(f"{p}: {all_data.get('preprocessing').get(p)}")
    print("-------------------- Feature extraction --------------------")
    for p in (all_features.get('fe_info').keys()):
        print(f"{p}: {all_features.get('fe_info').get(p)}")

def auc_datasizes_multisample(all_data, all_features, sizes = np.arange(90,900,90), 
                              #step_size = 90, 
                              tracker=False):
    """Compute AUC for different dataset sizes - use multiple samples for every dataset size"""

    X = all_features.get('features')
    X = np.reshape(X, (X.shape[0],-1))
    print(X.shape)
    n_epochs_total = len(X)
    start = 0
    #stop = n_epochs_total
    stop = 60696 # until and incl s10
    print(f"Total epochs: {n_epochs_total}")
    print(f"Sizes: {sizes}")
    print(f"Range of data to sample from: [{start} : {stop}] (in # epochs)")

    trials=all_data.get('trials')
    y = [(1 if event > 107 else 0) for trial in trials for iteration in trial for event in iteration.events[:,2]]            
    y = np.array(y) # conversion to np array is maybe not even needed
    print(y.shape)
    
    scores = np.zeros(len(sizes))
    nch = (all_data.get('trials')[0][0]).info["nchan"]
    

    for s,size in enumerate(sizes):
        print(size)
        step = size
        size_scores = []

        for j in range(start,stop,step):
            btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)

            if tracker:
                print(f"\t interval: [{start+j},{start+j+size}] (out of [{start},{stop}])")

                size_scores.append(cross_val_score(btlda, X[start+j:start+j+size, :], y[start+j:start+j+size], cv=4, scoring='roc_auc').mean())

        print("FINAL scores of this size: ",size_scores)     
        scores[s] = np.mean(size_scores) 
        print("mean: ", np.mean(size_scores))  

    return scores       