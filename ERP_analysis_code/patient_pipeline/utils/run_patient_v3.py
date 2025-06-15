# everything here is v2 because it uses the new time intervals for feature extraction (13-06-2025)
# time intervals: [0.1, 0.15, ..., 0.75, 0.8]

from utils.online_simulation import online_simulation, online_window_simulation_v3, online_window_simulation_v5, online_cc_simulation
from utils.preprocessing import load_session_chached, merge_sessions
from utils.feature_extraction import merge_features, load_features_chached_v2
import numpy as np
import os

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
def run_patient_online_sessions_CC(patient, last_session_nr, calibration_selection, UC_mean, UC_cov):
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
    online_result,old_clf = online_cc_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_cc_s{first_online}.log", UC_mean=UC_mean, UC_cov=UC_cov)
    print(f"OLD CLF: {old_clf}")
    performances.update({f"p{patient}_cc_s{first_online}":online_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        # data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        # features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")


        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        online_result,new_clf = online_cc_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_cc_s{i}.log", clf=old_clf, UC_mean=UC_mean, UC_cov=UC_cov)
        old_clf = new_clf

        performances.update({f"p{patient}_cc_s{i}":online_result})

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

