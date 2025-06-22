import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from toeplitzlda.classification import ToeplitzLDA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# utils functions
from utils.preprocessing import load_session_chached, merge_sessions
from utils.feature_extraction_v2 import epoch_vectorizer_channelprime

# Turn off warnings (that most likely occur from ToeplitzLDA)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')

def get_results_800ms_btlda():
    """Run cross validation over 4 time interval sizes ranging from 0.1-0.8 s on all patients"""

    all_patient_results = dict() 

    # for all patients
    for i in range(1,10):
        print(f"Patient {i} -----------------------------------")
        # load data of sessions 1 and 2
        data_s1 = load_session_chached(f"B:/anonymized_data/P0{i}a/P{i}_S1/anonymized", 
                                selection = "6D_long_350", 
                                discard_channels=True)
        data_s2 = load_session_chached(f"B:/anonymized_data/P0{i}a/P{i}_S2/anonymized", 
                                    selection = "6D_long_350", 
                                    discard_channels=True)
        data_s12 = merge_sessions(data_s1, data_s2)
        trials = data_s12.get('trials')

        # perform nested cv
        outer_results = nested_cv(trials)

        # store results
        all_patient_results[f'P{i}_6D_long_350'] = outer_results

    # patient 10:
    print(f"Patient {10} -----------------------------------")
    # load data of sessions 1 and 2
    data_s1 = load_session_chached(f"B:/anonymized_data/P10a/P10_S1/anonymized", 
                                selection = "6D_long_350", 
                                discard_channels=True)
    data_s2 = load_session_chached(f"B:/anonymized_data/P10a/P10_S2/anonymized", 
                                selection = "6D_long_350", 
                                discard_channels=True)
    data_s12 = merge_sessions(data_s1, data_s2)
    trials = data_s12.get('trials')

    # perform nested cv
    outer_results = nested_cv(trials)

    # store results
    all_patient_results[f'P10_6D_long_350'] = outer_results

    results = np.zeros((10,4))
    p_nr = 0
    for p in all_patient_results.keys():
        patient = all_patient_results.get(p)
        ival_nr = 0
        for time_ival in patient.keys():
            print("{:0.2f}: {}".format((1/len(time_ival[:]))*23,(patient.get(time_ival))))
            print(np.mean(patient.get(time_ival)))
            results[p_nr, ival_nr] = np.mean(patient.get(time_ival))
            ival_nr += 1
        p_nr += 1


# change range of arange from 0.1,0.51 to 0.1,0.81 (bc P600 is related to language)
def nested_cv(trials):
    """Cross validation to find optimal time intervals"""
    # Time intervals
    print("Time intervals:")
    ival_bounds_100 = np.arange(0.1,0.81,0.1) # [0.1 0.2 0.3 0.4...
    print(f"100 ms: \n{ival_bounds_100}")
    ival_bounds_50 = np.arange(0.1,0.81,0.05) # [0.1  0.15 0.2  0.25 0.3...
    print(f"50 ms: \n{ival_bounds_50}")
    ival_bounds_20 = np.arange(0.1,0.81,0.02) # [0.1  0.12 0.14 ... 
    print(f"20 ms: \n{ival_bounds_20}")
    ival_bounds_10 = np.arange(0.12,0.8,0.01) # [0.12 0.13 0.14 ...
    print(f"10 ms: \n{ival_bounds_10}")

    # Nested cv settings
    ival_grid = [
                ival_bounds_100, 
                ival_bounds_50, 
                ival_bounds_20, 
                ival_bounds_10
                ]
    n_trials = len(trials)
    K_folds = 4
    fold_size = int(n_trials/K_folds)

    print(f"\nn_trials: {len(trials)}")
    print(f"Doing a {K_folds}-fold cross-vaidation:")
    print(f"There are {K_folds} folds each of size: {fold_size}")

    print("\nPerforming cross-validation...")

    ival_counter = 0
    ival_list = [100, 50, 20, 10]
    all_ival_scores = dict()

    for ival in ival_grid:
        outer_kf = KFold(n_splits=K_folds, shuffle=False)
        scores = np.zeros(4)
        nch = (trials[0][0]).info["nchan"]
        fold_counter = 0

        for train_idx, test_idx in outer_kf.split(trials):

            X_train = [trials[i] for i in train_idx]  # | |x|x|x| for fold 1, |x| |x|x| for fold 2, ...
            X_test = [trials[i] for i in test_idx]    # |x| | | | for fold 1, | |x| | | for fold 2, ...

            # OUTER TEST using best ival
            X_train_complete, y_train_complete = epoch_vectorizer_channelprime(X_train,ival)
            X_test_complete, y_test_complete = epoch_vectorizer_channelprime(X_test, ival)

            # classifier
            clf_btlda_final = make_pipeline(
                ToeplitzLDA(n_channels=nch),
            )
            clf_btlda_final.fit(X_train_complete,y_train_complete)
            y_scores_test = clf_btlda_final.decision_function(X_test_complete)

            fold_score = roc_auc_score(y_test_complete, y_scores_test)
            scores[fold_counter] = fold_score
            fold_counter+=1

        print(f"\n Ival {ival_list[ival_counter]} ms")
        print(f"All folds scores: {scores}")
        print(f"Average AUC over all folds: {np.mean(scores):.4f}")
        all_ival_scores.update({f"{ival}":scores})
        ival_counter+=1
        
    return all_ival_scores

def get_results_1000ms_btlda():
    """Run cross validation over 4 time interval sizes ranging from 0.1-1.0 s on all patients"""

    all_patient_results_1s = dict() 

    # for all patients
    for i in range(1,10):
        print(f"Patient {i} -----------------------------------")
        # load data of sessions 1 and 2
        data_s1 = load_session_chached(f"B:/anonymized_data/P0{i}a/P{i}_S1/anonymized", 
                                selection = "6D_long_350", 
                                discard_channels=True)
        data_s2 = load_session_chached(f"B:/anonymized_data/P0{i}a/P{i}_S2/anonymized", 
                                    selection = "6D_long_350", 
                                    discard_channels=True)
        data_s12 = merge_sessions(data_s1, data_s2)
        trials = data_s12.get('trials')

        # perform nested cv
        outer_results = nested_cv_extended(trials)

        # store results
        all_patient_results_1s[f'P{i}_6D_long_350'] = outer_results

    # patient 10:
    print(f"Patient {10} -----------------------------------")
    # load data of sessions 1 and 2
    data_s1 = load_session_chached(f"B:/anonymized_data/P10a/P10_S1/anonymized", 
                                selection = "6D_long_350", 
                                discard_channels=True)
    data_s2 = load_session_chached(f"B:/anonymized_data/P10a/P10_S2/anonymized", 
                                selection = "6D_long_350", 
                                discard_channels=True)
    data_s12 = merge_sessions(data_s1, data_s2)
    trials = data_s12.get('trials')

    # perform nested cv
    outer_results = nested_cv_extended(trials)

    # store results
    all_patient_results_1s[f'P10_6D_long_350'] = outer_results

# change range of arange from 0.1,0.51 to 0.1,1.01 (bc P600 is related to language)
def nested_cv_extended(trials):
    """Cross validation to find optimal time intervals, extended to 1000 ms"""
    # Time intervals
    print("Time intervals:")
    ival_bounds_100 = np.arange(0.1,1.01,0.1) # [0.1 0.2 0.3 0.4...
    print(f"100 ms: \n{ival_bounds_100}")
    ival_bounds_50 = np.arange(0.1,1.01,0.05) # [0.1  0.15 0.2  0.25 0.3...
    print(f"50 ms: \n{ival_bounds_50}")
    ival_bounds_20 = np.arange(0.1,1.01,0.02) # [0.1  0.12 0.14 ... 
    print(f"20 ms: \n{ival_bounds_20}")
    ival_bounds_10 = np.arange(0.12,1.01,0.01) # [0.12 0.13 0.14 ...
    print(f"10 ms: \n{ival_bounds_10}")

    # Nested cv settings
    ival_grid = [
                ival_bounds_100, 
                ival_bounds_50, 
                ival_bounds_20, 
                ival_bounds_10
                ]
    n_trials = len(trials)
    K_folds = 4
    fold_size = int(n_trials/K_folds)

    print(f"\nn_trials: {len(trials)}")
    print(f"Doing a {K_folds}-fold cross-vaidation:")
    print(f"There are {K_folds} folds each of size: {fold_size}")

    print("\nPerforming cross-validation...")

    ival_counter = 0
    ival_list = [100, 50, 20, 10]
    all_ival_scores = dict()

    for ival in ival_grid:
        outer_kf = KFold(n_splits=K_folds, shuffle=False)
        scores = np.zeros(4)
        nch = (trials[0][0]).info["nchan"]
        fold_counter = 0

        for train_idx, test_idx in outer_kf.split(trials):

            X_train = [trials[i] for i in train_idx]  # | |x|x|x| for fold 1, |x| |x|x| for fold 2, ...
            X_test = [trials[i] for i in test_idx]    # |x| | | | for fold 1, | |x| | | for fold 2, ...

            # OUTER TEST using best ival
            X_train_complete, y_train_complete = epoch_vectorizer_channelprime(X_train,ival)
            X_test_complete, y_test_complete = epoch_vectorizer_channelprime(X_test, ival)

            # classifier
            clf_btlda_final = make_pipeline(
                ToeplitzLDA(n_channels=nch),
            )
            clf_btlda_final.fit(X_train_complete,y_train_complete)
            y_scores_test = clf_btlda_final.decision_function(X_test_complete)

            fold_score = roc_auc_score(y_test_complete, y_scores_test)
            scores[fold_counter] = fold_score
            fold_counter+=1

        print(f"\n Ival {ival_list[ival_counter]} ms")
        print(f"All folds scores: {scores}")
        print(f"Average AUC over all folds: {np.mean(scores):.4f}")
        all_ival_scores.update({f"{ival}":scores})
        ival_counter+=1
        
    return all_ival_scores


# Shrinkage LDA -----------------------------------------------------------------------------------

def get_results_800ms_slda():
    """Run cross validation with SLDA over 4 time interval sizes ranging from 0.1-0.8 s on all patients"""

    all_patient_results_slda = dict() 

    # for all patients
    for i in range(1,11):
        print(f"Patient {i} -----------------------------------")
        # load data of sessions 1 and 2
        if i<10:
            data_s1 = load_session_chached(f"B:/anonymized_data/P0{i}a/P{i}_S1/anonymized", 
                                    selection = "6D_long_350", 
                                    discard_channels=True)
            data_s2 = load_session_chached(f"B:/anonymized_data/P0{i}a/P{i}_S2/anonymized", 
                                        selection = "6D_long_350", 
                                        discard_channels=True)
            
        else:
            data_s1 = load_session_chached(f"B:/anonymized_data/P{i}a/P{i}_S1/anonymized", 
                                    selection = "6D_long_350", 
                                    discard_channels=True)
            data_s2 = load_session_chached(f"B:/anonymized_data/P{i}a/P{i}_S2/anonymized", 
                                        selection = "6D_long_350", 
                                        discard_channels=True)   
            
        data_s12 = merge_sessions(data_s1, data_s2)
        trials = data_s12.get('trials')

        # perform nested cv
        outer_results = nested_cv_slda(trials)

        # store results
        all_patient_results_slda[f'P{i}_6D_long_350'] = outer_results

# range ival 0.1-0.81
def nested_cv_slda(trials):
    """Cross validation to find optimal time intervals - ranging to 800 ms - SLDA instead of BT-LDA"""
    # Time intervals
    print("Time intervals:")
    ival_bounds_100 = np.arange(0.1,0.81,0.1) # [0.1 0.2 0.3 0.4 0.5] (4)
    print(f"100 ms: \n{ival_bounds_100}")
    ival_bounds_50 = np.arange(0.1,0.81,0.05) # [0.1  0.15 0.2  0.25 0.3  0.35 0.4  0.45 0.5] (8)
    print(f"50 ms: \n{ival_bounds_50}")
    ival_bounds_20 = np.arange(0.1,0.81,0.02) # [0.1  0.12 0.14 ... 0.46 0.48 0.5] (20)
    print(f"20 ms: \n{ival_bounds_20}")
    ival_bounds_10 = np.arange(0.12,0.81,0.01) # [0.12 0.13 0.14 ... 0.48 0.49 0.5] # might be changed
    print(f"10 ms: \n{ival_bounds_10}")

    # Nested cv settings
    ival_grid = [
                ival_bounds_100, 
                ival_bounds_50, 
                ival_bounds_20, 
                ival_bounds_10
                ]
    n_trials = len(trials)
    K_folds = 4
    fold_size = int(n_trials/K_folds)

    print(f"\nn_trials: {len(trials)}")
    print(f"Doing a {K_folds}-fold cross-vaidation:")
    print(f"There are {K_folds} folds each of size: {fold_size}")

    print("\nPerforming cross-validation...")

    ival_counter = 0
    ival_list = [100, 50, 20, 10]
    all_ival_scores = dict()

    for ival in ival_grid:
        outer_kf = KFold(n_splits=K_folds, shuffle=False)
        scores = np.zeros(4)
        fold_counter = 0

        for train_idx, test_idx in outer_kf.split(trials):

            X_train = [trials[i] for i in train_idx]  # | |x|x|x| for fold 1, |x| |x|x| for fold 2, ...
            X_test = [trials[i] for i in test_idx]    # |x| | | | for fold 1, | |x| | | for fold 2, ...

            # OUTER TEST using best ival
            X_train_complete, y_train_complete = epoch_vectorizer_channelprime(X_train,ival)
            X_test_complete, y_test_complete = epoch_vectorizer_channelprime(X_test, ival)

            # classifier
            slda = make_pipeline(LDA(solver='lsqr', shrinkage='auto'),) # SLDA

            slda.fit(X_train_complete,y_train_complete)
            y_scores_test = slda.decision_function(X_test_complete)

            fold_score = roc_auc_score(y_test_complete, y_scores_test)
            scores[fold_counter] = fold_score
            fold_counter+=1

        print(f"\n Ival {ival_list[ival_counter]} ms")
        print(f"All folds scores: {scores}")
        print(f"Average AUC over all folds: {np.mean(scores):.4f}")
        all_ival_scores.update({f"{ival}":scores})
        ival_counter+=1
        
    return all_ival_scores
