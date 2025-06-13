# everything here is v2 because it uses the new time intervals for feature extraction (13-06-2025)
# time intervals: [0.1, 0.15, ..., 0.75, 0.8]

from utils.online_simulation import online_simulation, online_window_simulation_v1, online_transfer_simulation_v2
from utils.preprocessing import load_session_chached, merge_sessions
from utils.offline_evaluation import compare_auc_single_trial_interval, compute_auc_with_cv
from utils.feature_extraction_v2 import merge_features, load_or_extract_markers, load_features_chached_v2
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
def run_patient_online_sessions_static_v2(patient, last_session_nr, calibration_selection, starter_conditions=None):
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

    static_result = online_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_static_s{first_online}_v2.log")
    performances.update({f"p{patient}_static_s{first_online}_v2":static_result})

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
        static_result = online_simulation(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_static_s{i}_v2.log", title_text=plot_title_text)

        performances.update({f"p{patient}_static_s{i}_v2":static_result})

    return performances

# Window v2
def run_patient_online_sessions_window_v2(patient, last_session_nr, calibration_selection):
    # adaptive window, v1: window size is previous session

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

    window_result = online_window_simulation_v1(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{first_online}_v2.log")
    performances.update({f"p{patient}_window_v2_s{first_online}":window_result})

    # Rest of online sessions

    for i in range(first_online+1,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        features_train = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i-1}_anonymized.pkl")
        features_test = load_features_chached_v2(f"B:_anonymized_data_P{patient_string}a_P{patient}_S{i}_anonymized.pkl")

        plot_title_text = f"patient {patient} session {i}"

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        window_result = online_window_simulation_v1(data_train, data_test, features_train, features_test, log_process=f"p{patient}_online_window_s{i}_v2.log", title_text=plot_title_text)

        performances.update({f"p{patient}_window_v2_s{i}":window_result})

    return performances

