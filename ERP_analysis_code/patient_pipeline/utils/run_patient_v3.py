# everything here is v3 because it uses the new time intervals for feature extraction (13-06-2025)
# time intervals: [0.1, 0.15, ..., 0.75, 0.8]

from utils.online_simulation import online_simulation, online_window_simulation_v3, online_window_simulation_v5, online_cc_simulation, online_transfer_simulation_v2
from utils.preprocessing import load_session_chached, merge_sessions
from utils.feature_extraction_v2 import merge_features, load_features_chached_v2
import numpy as np
import os

# Transfer
def run_patient_online_sessions_transfer_v2(patient, last_session_nr, calibration_selection):
    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    transfer_result = online_transfer_simulation_v2(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_transfer_s{first_online}_v2.log")
    performances.update({f"p{patient}_transfer_s{first_online}_v2":transfer_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")

        plot_title_text = f"patient {patient} session {i}"

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        transfer_result = online_transfer_simulation_v2(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_transfer_s{i}_v2.log", title_text=plot_title_text)

        performances.update({f"p{patient}_transfer_s{i}_v2":transfer_result})

    return performances

# Static
def run_patient_online_sessions_static(patient, last_session_nr, calibration_selection, starter_conditions=None):
    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if starter_conditions is not None:
        changing_conditions = True
    else:
        changing_conditions = False   

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    static_result = online_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_static_s{first_online}_v1.log")
    performances.update({f"p{patient}_static_s{first_online}":static_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        #  only change training data if conditions change, otherwise just keep old training data
        if changing_conditions:
            if i in starter_conditions:
                new_selection = starter_conditions.get(i)
                print(f"Changing condition! new selection: {new_selection} from session {i} on")
                new_selection_dc =  f"{new_selection}_dc"

                data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
                data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
                data_s1 = load_session_chached(data_path_s1, selection = new_selection, discard_channels=True)
                data_s2 = load_session_chached(data_path_s2, selection = new_selection, discard_channels=True)
                data_train = merge_sessions(data_s1, data_s2)
                features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{new_selection_dc}.pkl")
                features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{new_selection_dc}.pkl")
                features_train = merge_features(features_s1, features_s2)
            
                if patient == 8:
                    data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = new_selection, discard_channels=True)
                    data_train = merge_sessions(data_train, data_s3)
                    features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{new_selection_dc}.pkl")
                    features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")
        plot_title_text = f"patient {patient} session {i}"

        # 3. Online simulation static fixed (trained on session 1+2 following static_protocol)
        static_result = online_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_static_s{i}_v1.log", title_text=plot_title_text)

        performances.update({f"p{patient}_static_s{i}":static_result})

    return performances

# Aphasia adaptive sLDA [1]
#
# [1] M. Musso et al., “Aphasia recovery by language training using a brain–computer interface: a proof-of-concept study,” Brain Communications, vol. 4, no. 1, p. fcac008, Feb. 2022, doi: 10.1093/braincomms/fcac008.
def run_patient_online_sessions_aphasia_slda(patient, 
                                             last_session_nr, 
                                             calibration_selection, 
                                             starter_conditions=None, 
                                             UC_mean=0.005, 
                                             UC_cov=0.001):
    # UC_cov = 0.001 for the global cov matrix

    performances = dict()

    if starter_conditions is not None:
        changing_conditions = True
    else:
        changing_conditions = False  

    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")
    
    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    # store results in folder adaptive_slda
    log_dir = f"adaptive_slda/logs"
    os.makedirs(log_dir, exist_ok=True)

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    # Run first online simulation
    log_path = os.path.join(log_dir, f"p{patient}_adaptive_slda_s{first_online}.log")

    online_result,old_clf = online_cc_simulation(data_train, data_test, features_train, features_test, 
                                                 log_process=log_path, 
                                                 UC_mean=UC_mean, UC_cov=UC_cov, adaptive_slda=True)
    performances.update({f"p{patient}_adaptive_slda_s{first_online}":online_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        new_classifier = False

        #  only change training data if conditions change, otherwise just keep old training data
        if changing_conditions:
            # Check if we are facing a session with a new condition
            if i in starter_conditions:
                old_clf=None # if yes, then remove the old classifier for the convex combination, so a new classifier is trained
                new_selection = starter_conditions.get(i)
                print(f"Changing condition! new selection: {new_selection} from session {i} on")
                new_selection_dc =  f"{new_selection}_dc"

                data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
                data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
                data_s1 = load_session_chached(data_path_s1, selection = new_selection, discard_channels=True)
                data_s2 = load_session_chached(data_path_s2, selection = new_selection, discard_channels=True)
                data_train = merge_sessions(data_s1, data_s2)
                features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{new_selection_dc}.pkl")
                features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{new_selection_dc}.pkl")
                features_train = merge_features(features_s1, features_s2)
            
                if patient == 8:
                    data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = new_selection, discard_channels=True)
                    data_train = merge_sessions(data_train, data_s3)
                    features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{new_selection_dc}.pkl")
                    features_train = merge_features(features_train, features_s3)

        
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")

        # 3. Online simulation
        log_path = os.path.join(log_dir, f"p{patient}_adaptive_slda_s{i}.log")

        online_result,new_clf = online_cc_simulation(data_train, data_test, features_train, features_test, 
                                                     log_process=log_path, 
                                                     clf=old_clf, UC_mean=UC_mean, UC_cov=UC_cov, adaptive_slda=True)
        old_clf = new_clf

        performances.update({f"p{patient}_adaptive_slda_s{i}":online_result})

    return performances

# CC for all UC pairs
def run_patient_online_sessions_CC_UC_pairs(patient, last_session_nr, calibration_selection, UC_mean, UC_cov, version=None):
    # CC: ivals 0.1-0.81, 50ms steps

    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if version is None:
        version = ""
        version_string=""
    else:
        version_string=f"_v{version}"    

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    # store results in folder results_UC
    log_dir = f"results_UC/v{version}/logs"
    os.makedirs(log_dir, exist_ok=True)

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    # Run first online simulation
    log_path = os.path.join(log_dir, f"p{patient}_online_cc{version_string}_s{first_online}.log")

    online_result,old_clf = online_cc_simulation(data_train, data_test, features_train, features_test, 
                                                 log_process=log_path, 
                                                 UC_mean=UC_mean, UC_cov=UC_cov)
    performances.update({f"p{patient}_cc_s{first_online}":online_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        
        # 1. Only load the runs of the previous session as training data
        # data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        # features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")


        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        log_path = os.path.join(log_dir, f"p{patient}_online_cc{version_string}_s{i}.log")

        online_result,new_clf = online_cc_simulation(data_train, data_test, features_train, features_test, 
                                                     log_process=log_path, 
                                                     clf=old_clf, UC_mean=UC_mean, UC_cov=UC_cov)
        old_clf = new_clf

        performances.update({f"p{patient}_cc_s{i}":online_result})

    return performances


# CC
def run_patient_online_sessions_CC(patient, last_session_nr, calibration_selection, UC_mean, UC_cov, version=""):
    # CC: ivals 0.1-0.81, 50ms steps

    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    # Run first online simulation
    if version == "":
        version_string = "_v00"  
    else:
        version_string = f"_v{version}"

    online_result,old_clf = online_cc_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_cc{version_string}_s{first_online}.log", UC_mean=UC_mean, UC_cov=UC_cov)
    print(f"OLD CLF: {old_clf}")
    performances.update({f"p{patient}_cc{version_string}_s{first_online}":online_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        # data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        # features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")


        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        
        online_result,new_clf = online_cc_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_cc{version_string}_s{i}.log", clf=old_clf, UC_mean=UC_mean, UC_cov=UC_cov)
        old_clf = new_clf

        performances.update({f"p{patient}_cc{version_string}_s{i}":online_result})

    return performances

# Window v5
def run_patient_online_sessions_window_v5(patient, last_session_nr, calibration_selection):
    # adaptive window, v5: window size is WINDOW SIZE S - correct sw (not gw) - ivals 0.1-0.81, 50ms steps

    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    window_result = online_window_simulation_v5(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{first_online}_v5.log")
    performances.update({f"p{patient}_window_v5_s{first_online}":window_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")

        plot_title_text = f"patient {patient} session {i}"

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        window_result = online_window_simulation_v5(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{i}_v5.log", title_text=plot_title_text)

        performances.update({f"p{patient}_window_v5_s{i}":window_result})

    return performances





# Window v4
def run_patient_online_sessions_window_v4(patient, last_session_nr, calibration_selection):
    # adaptive window, v4: window size is previous session - correct sw (not gw) - ivals 0.1-0.81, 50ms steps

    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached_v2(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    window_result = online_window_simulation_v3(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{first_online}_v4.log")
    performances.update({f"p{patient}_window_v4_s{first_online}":window_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")

        plot_title_text = f"patient {patient} session {i}"

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        window_result = online_window_simulation_v3(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{i}_v4.log", title_text=plot_title_text)

        performances.update({f"p{patient}_window_v4_s{i}":window_result})

    return performances



# Window v3
def run_patient_online_sessions_window_v3(patient, last_session_nr, calibration_selection):
    # adaptive window, v3: window size is previous session - correct sw (not gw) - ivals 0.1-0.51, 50ms steps

    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### First online session --------------------------------------------------------
    #Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    calibration_selection_dc =  f"{calibration_selection}_dc"

    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    features_s1 = load_features_chached(f"B:_anonymized_data_P{patient_string}a_P{patient}_S1_anonymized{calibration_selection_dc}.pkl")
    features_s2 = load_features_chached(f"B:_anonymized_data_P{patient_string}a_P{patient}_S2_anonymized{calibration_selection_dc}.pkl")
    features_train = merge_features(features_s1, features_s2)

    if patient == 8:
        data_s3 = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized", selection = calibration_selection, discard_channels=True)
        data_train = merge_sessions(data_train, data_s3)
        features_s3 = load_features_chached(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized{calibration_selection_dc}.pkl")
        features_train = merge_features(features_train, features_s3)

        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S4/anonymized")
        features_test = load_features_chached(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S4_anonymized.pkl")

        first_online = 4

    else:
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
        features_test = load_features_chached(fr"B:_anonymized_data_P{patient_string}a_P{patient}_S3_anonymized.pkl")
        first_online = 3

    window_result = online_window_simulation_v3(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{first_online}_v3.log")
    performances.update({f"p{patient}_window_v3_s{first_online}":window_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_train = load_features_chached(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")

        plot_title_text = f"patient {patient} session {i}"

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        window_result = online_window_simulation_v3(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{i}_v3.log", title_text=plot_title_text)

        performances.update({f"p{patient}_window_v3_s{i}":window_result})

    return performances

