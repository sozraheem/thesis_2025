from utils.online_simulation import online_simulation, online_adaptation_simulation_sw
from utils.preprocessing import load_session_chached, merge_sessions
from utils.offline_evaluation import compare_auc_single_trial_interval, compute_auc_with_cv
import numpy as np
import os

def run_patient_simulation(patient, last_session_nr, calibration_selection, include_offline_performance = True, ival_bounds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])):
    """
    Run online simulation on all sessions of a patient. Store results of transfer ("transfer fixed") and adaptive sliding window
    Note: this does not work for patient 8, since it seems that session 3 for this patient was also a calibration session
    
    Parameters:
    - patient (int): patient number
    - last_session_nr (int)
    - calibration_selection (string): selected files to load for calibration data. E.g. "6D_long_350" or "6D_short_250"
    - include_offline_performance (boolean): True if offline performance for the calibration data should be printed, False otherwise.

    Output: no returns. This function prints resutls and writes to multiple .log files

    Example usage:
    > performances = run_patient_simulation(patient=7, last_session_nr=5, calibration_selection="6D_long_350", include_offline_performance=True)
    """
    
    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")
    # if include_offline_performance:
    #     directory_name = f"p{patient}_offline"
    #     try:
    #         os.mkdir(directory_name)
    #     except Exception as e:
    #         print(f"An error occured while making the offline directory: {e}")    

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### Session 3 --------------------------------------------------------
    # Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
    plot_title_text = f"patient {patient} session 3"

    # Offline performance
    if include_offline_performance:
        trials_train = data_train.get("trials")
        #clf_ival_boundaries = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        clf_ival_boundaries = ival_bounds
        compare_auc_single_trial_interval(trials_train, only_auc = True, ival_bounds = clf_ival_boundaries, plot_roc_curves=True, title_text = plot_title_text)
        compute_auc_with_cv(trials_train, ival_bounds=clf_ival_boundaries, cv_folds=4, show_mean=True, show_folds=False, title_text = plot_title_text)

    # Online performance
    transfer_result = online_simulation(data_train, data_test, ival_bounds=ival_bounds, log_process=f"p{patient}_online_transfer_s3.log", title_text =plot_title_text)
    adaptive_sw_result = online_adaptation_simulation_sw(data_train, data_test, ival_bounds=ival_bounds, log_process=f"p{patient}_online_adaptive_sw_s3.log", title_text = plot_title_text)
    performances.update({f"p{patient}_transfer_s3":transfer_result})
    performances.update({f"p{patient}_adaptive_sw_s3":adaptive_sw_result})

    ### Sessions 4 - last_online_session -----------------------------------------------------
    for i in range(4,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        plot_title_text = f"patient {patient} session {i}"
        
        # 2. Evaluate offline performance 
        if include_offline_performance:
            trials_train = data_train.get("trials")
            #clf_ival_boundaries = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            clf_ival_boundaries = ival_bounds
            compare_auc_single_trial_interval(trials_train, start=0, stop=None, test_size=0.2, only_auc = True, ival_bounds = clf_ival_boundaries, plot_roc_curves=True)
            compute_auc_with_cv(trials_train, start=0, stop=None, ival_bounds=clf_ival_boundaries, cv_folds=4, show_mean=True, show_folds=False)

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        transfer_result = online_simulation(data_train, data_test, ival_bounds=ival_bounds, log_process=f"p{patient}_online_transfer_s{i}.log", title_text=plot_title_text)
        
        # 4. Online simulation with sliding window adaptation (trained on session i-1 - applied on session i + adaptation)
        adaptive_sw_result = online_adaptation_simulation_sw(data_train, data_test, ival_bounds=ival_bounds, log_process=f"p{patient}_online_adaptive_sw_s{i}.log", title_text=plot_title_text)

        performances.update({f"p{patient}_transfer_s{i}":transfer_result})
        performances.update({f"p{patient}_adaptive_sw_s{i}":adaptive_sw_result})

    return performances    

def run_patient_simulation_static_gw(patient, last_session_nr, calibration_selection, include_offline_performance = True):
    """
    Run online simulation on all sessions of a patient. Store results of static ("static fixed") and adaptive growing window
    Note: this does not work for patient 8, since it seems that session 3 for this patient was also a calibration session
    
    Parameters:
    - patient (int): patient number
    - last_session_nr (int)
    - calibration_selection (string): selected files to load for calibration data. E.g. "6D_long_350" or "6D_short_250"
    - include_offline_performance (boolean): True if offline performance for the calibration data should be printed, False otherwise.

    Output: no returns. This function prints resutls and writes to multiple .log files

    Example usage:
    > performances = run_patient_simulation_static(patient=7, last_session_nr=5, calibration_selection="6D_long_350", include_offline_performance=False)
    """
    
    if not isinstance(calibration_selection, str):
        raise ValueError("calibration_selection should be a string")
    if not (calibration_selection in ["6D_long_350", "6D_short_250"]):
        raise ValueError("calibration_selection can be either 6D_long_350 or 6D_short_250")
    # if include_offline_performance:
    #     directory_name = f"p{patient}_offline"
    #     try:
    #         os.mkdir(directory_name)
    #     except Exception as e:
    #         print(f"An error occured while making the offline directory: {e}")    

    if patient<10:
        patient_string = f"0{patient}"
    else:
        patient_string = f"{patient}"

    performances = dict()

    ### Session 3 --------------------------------------------------------
    # Calibration data: sessions 1 and 2 (only runs with same condition as S3)
    data_path_s1 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S1/anonymized"
    data_path_s2 = f"B:/anonymized_data/P{patient_string}a/P{patient}_S2/anonymized"
    data_s1 = load_session_chached(data_path_s1, selection = calibration_selection, discard_channels=True)
    data_s2 = load_session_chached(data_path_s2, selection = calibration_selection, discard_channels=True)
    data_train = merge_sessions(data_s1, data_s2)
    data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S3/anonymized")
    plot_title_text = f"patient {patient} session 3"

    # Offline performance
    if include_offline_performance:
        trials_train = data_train.get("trials")
        clf_ival_boundaries = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        compare_auc_single_trial_interval(trials_train, only_auc = True, ival_bounds = clf_ival_boundaries, plot_roc_curves=True, title_text = plot_title_text)
        compute_auc_with_cv(trials_train, ival_bounds=clf_ival_boundaries, cv_folds=4, show_mean=True, show_folds=False, title_text = plot_title_text)

    # Online performance
    transfer_result = online_simulation(data_train, data_test, log_process=f"p{patient}_online_transfer_s3.log", title_text =plot_title_text)
    adaptive_sw_result = online_adaptation_simulation_sw(data_train, data_test, log_process=f"p{patient}_online_adaptive_gw_s3.log", title_text = plot_title_text, growing=True)
    performances.update({f"p{patient}_static_s3":transfer_result})
    performances.update({f"p{patient}_adaptive_gw_s3":adaptive_sw_result})

    ### Sessions 4 - last_online_session -----------------------------------------------------
    for i in range(4,last_session_nr):
        # 1. Only load the runs of the previous session as training data
        # data_train = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i-1}/anonymized")
        data_test = load_session_chached(f"B:/anonymized_data/P{patient_string}a/P{patient}_S{i}/anonymized")
        plot_title_text = f"patient {patient} session {i}"
        
        # 2. Evaluate offline performance 
        if include_offline_performance:
            trials_train = data_train.get("trials")
            clf_ival_boundaries = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            compare_auc_single_trial_interval(trials_train, start=0, stop=None, test_size=0.2, only_auc = True, ival_bounds = clf_ival_boundaries, plot_roc_curves=True)
            compute_auc_with_cv(trials_train, start=0, stop=None, ival_bounds=clf_ival_boundaries, cv_folds=4, show_mean=True, show_folds=False)

        # 3. Online simulation transfer fixed (trained on session i-1 - applied on session i)
        transfer_result = online_simulation(data_train, data_test, log_process=f"p{patient}_online_static_s{i}.log", title_text=plot_title_text)
        
        # 4. Online simulation with sliding window adaptation (trained on session i-1 - applied on session i + adaptation)
        adaptive_sw_result = online_adaptation_simulation_sw(data_train, data_test, log_process=f"p{patient}_online_adaptive_gw_s{i}.log", title_text=plot_title_text, growing=True)

        performances.update({f"p{patient}_static_s{i}":transfer_result})
        performances.update({f"p{patient}_adaptive_gw_s{i}":adaptive_sw_result})

    return performances    