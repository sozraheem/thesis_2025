import mne
import numpy as np
from pathlib import Path
from utils.preprocessing import non_eeg_channels
import logging
from datetime import datetime

def log_patient_marker_information(patient_number, last_session_number, log_name, track_progress=False):
    """
    For each session, writes to a log file all common and odd markers across all runs. Also the filenames were the odd markers are found, are logged. 
    Note that this function currently only works for data loaded from the hard drive.

    Example usage:
    # Example 1
    > patient_number  = 1
    > last_session_number = 18
    > #patient_path = f"B:/anonymized_data/P0{patient_number}a"
    > log_patient_marker_information(patient_number=patient_number, last_session_number=last_session_number, log_name=f"test_p{patient_number}_info.log", track_progress=True)
    """

    assert isinstance(log_name, str), "Invalid value has been given for log_name. It should be a string."

    start_logging(log_name)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp}")
    logging.info(f"Patient {patient_number} marker information of all sessions 1-{last_session_number}")

    # TO DO: add content of _print_marker_information()

    if patient_number<10:
        patient_path = f"B:/anonymized_data/P0{patient_number}a"
    else:
        patient_path = f"B:/anonymized_data/P{patient_number}a"   

    string1 = "\nCommon markers across all runs, for each session"
    string2 = "\nMarkers that do not appear in all runs, for each session"
    odd_markers = []

    for session in range(1,last_session_number+1):
        if patient_path is None:
            data_path = f"data_p{patient_number}/P{patient_number}_S{session}/anonymized"
            if(track_progress):
                print(data_path)
        else:
            data_path = patient_path+f"/P{patient_number}_S{session}/anonymized"
            if(track_progress):
                print(data_path)

        x = (get_markers_of_session(data_path))
        string1 += f"\nSession {session}: {x[3]}"

        odd_markers.append(x[4])
        string2 += f"\nSession {session}: {"-" if x[4]==set() else x[4]}"

    logging.info(string1)  
    logging.info(string2)

    odd_markers_old = odd_markers
    odd_markers = np.unique([m for mrks in odd_markers for m in mrks])

    logging.info(f"\nSearching for the uncommon markers: {odd_markers}")
    logging.info("[filename]: [(marker, count)]")

    for session in range(1,last_session_number+1):
        if patient_path is None:
            data_path = f"data_p{patient_number}/P{patient_number}_S{session}/anonymized"
            if(track_progress):
                print(data_path)
        else:
            data_path = patient_path+f"/P{patient_number}_S{session}/anonymized"
            if(track_progress):
                print(data_path)
        filenames = get_session_filenames_with_markers(data_path, marker_list= odd_markers, track_progress=False)
        for f in filenames:
            if len(filenames.get(f))>0:
                logging.info(f"{f}: {filenames.get(f)}")

    logging.info("\nEnd of log file")

    close_logging()

    return odd_markers_old, odd_markers


def start_logging(log_file_name):
    # this was needed in order to create a log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file_name,
        encoding="utf-8",
        filemode="w", # 'a' to not overwrite current log, 'w' to overwrite. This setting can be changed later
        level=logging.DEBUG,
        format='%(message)s')

def close_logging():
    # close and remove all handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_markers_of_session(data_path, track_progress=False):
    """
    Get all unique markers of a session

    Output:
        (
        filenames, 
        marker_names, 
        all_marker_uniques, 
        sorted(markers_in_all_runs), 
        not_in_all_runs_contain
        )

    Example usage:
    > data_path = "data_p1/P1_S3/anonymized"
    > markers_tuple = get_markers_of_session(data_path) 
    """

    data_dir = Path.cwd() / data_path
    header_files = data_dir.glob("auditoryAphasia*.vhdr")

    filenames = []
    marker_names = dict() # this contains the marker information per marker
    all_marker_uniques = []
    not_all_runs_contain = list() # list with markers that are not found in all runs
    markers_in_all_runs = None

    for f in header_files:
        #raw_data = _load_and_preprocess_raw(f)
        raw_data = mne.io.read_raw_brainvision(f, misc=non_eeg_channels, preload=True)

        # To store metadata 
        parent1 = f.parent.name            # immediate parent ("anonymized")
        parent2 = f.parent.parent.name     # grandparent (e.g. "P1_S5")
        filenames.append(f"{parent2}/{parent1}/{f.name}")
        
        if(track_progress):
            print(f"Loaded {parent2}/{parent1}/{f.name}")

        evs = mne.events_from_annotations(raw_data)
        marker_names.update(evs[1])
        run_marker_uniques = set(evs[0][:,2])
        all_marker_uniques.append(run_marker_uniques)

        # Start with storing the markers of the first run
        if markers_in_all_runs is None:
            markers_in_all_runs = run_marker_uniques
        else:
            apart = run_marker_uniques - markers_in_all_runs
            not_all_runs_contain.append(apart)
            common = markers_in_all_runs & run_marker_uniques
            markers_in_all_runs = common

    all_marker_uniques = np.unique([mrks for run in all_marker_uniques for mrks in run])
    not_in_all_runs_contain = set(all_marker_uniques) - (markers_in_all_runs)

    return (
        filenames, 
        marker_names, 
        all_marker_uniques, 
        sorted(markers_in_all_runs), 
        not_in_all_runs_contain
        )

def print_patient_common_and_apart_markers(patient_number, last_session_number, patient_path = None):
    """
    Example output:
    --------------------------------- Session 1 ---------------------------------
    Common markers across all runs: [101, 102, 103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 200, 201, 202, 203, 204, 205, 255, 99999]
    Markers in some but not all runs: -
    --------------------------------- Session 2 ---------------------------------
    """
    for session in range(1,last_session_number+1):
        if patient_path is None:
            data_path = f"data_p{patient_number}/P{patient_number}_S{session}/anonymized"
        else:
            data_path = patient_path+f"/P{patient_number}_S{session}/anonymized"

    marker_tuple = list()
    for s in range(1,19): 
        print(f"--------------------------------- Session {s} ---------------------------------")
        data_path = f"data_p1/P1_S{s}/anonymized"
        marker_tuple = get_markers_of_session(data_path)
        print(f"Common markers across all runs: {marker_tuple[3]}")
        print(f"Markers in some but not all runs: {"-" if marker_tuple[4]==set() else marker_tuple[4]}")



# def get_markers_of_session(data_path, track_progress=False):
#     """
#     Get all unique markers of a session

#     Example usage:
#     > data_path = "data_p1/P1_S3/anonymized"
#     > markers_tuple = get_markers_of_session(data_path) 
#     """

#     data_dir = Path.cwd() / data_path
#     header_files = data_dir.glob("auditoryAphasia*.vhdr")
#     filenames = list()
#     marker_names = dict() # this contains the marker information per marker
#     all_marker_uniques = list()
#     all_runs_contain = list() # list with markers that are found in all runs
#     not_all_runs_contain = list() # list with markers that are not found in all runs
#     first_run = True

#     for f in header_files:
#         #raw_data = _load_and_preprocess_raw(f)
#         raw_data = mne.io.read_raw_brainvision(f, misc=non_eeg_channels, preload=True)

#         # To store metadata 
#         parent1 = f.parent.name            # immediate parent ("anonymized")
#         parent2 = f.parent.parent.name     # grandparent (e.g. "P1_S5")
#         filenames.append(f"{parent2}/{parent1}/{f.name}")
        
#         if(track_progress):
#             print(f"Loaded {parent2}/{parent1}/{f.name}")

#         evs = mne.events_from_annotations(raw_data)
#         marker_names.update(evs[1])
#         run_marker_uniques = np.unique(evs[0][:,2])
#         all_marker_uniques.append(run_marker_uniques)

#         if first_run:
#             first_run = False
#             all_runs_contain.append(np.unique(evs[0][:,2]))
#         else:
#             new_all_runs_contain = list()
#             for mrk in [run_marker_uniques]:
#                 if mrk in all_runs_contain:
#                     new_all_runs_contain.append(mrk)
#                 else:
#                     not_all_runs_contain.append(mrk)
        
#     all_marker_uniques = [mrks for run in all_marker_uniques for mrks in run]    
#     return (filenames, marker_names, np.unique(all_marker_uniques), all_runs_contain, not_all_runs_contain)

def get_markers_of_patient(patient_number, last_session_number, patient_path = None, track_progress = False, print_result=True):
    """
    For all sessions of a single patient: return all filenames, all marker information (dictionary) and all unique markers

    Example usage:
    > patient_nr = 9
    > last_session = 18
    > patient_path = f"B:/anonymized_data/P0{patient_nr}a"
    > patient_marker_info = get_markers_of_patient(patient_nr,last_session, patient_path=patient_path)
        
    """
    total_filenames = list()
    total_marker_names = dict()
    total_marker_uniques = list()

    for session in range(1,last_session_number+1):
        if patient_path is None:
            data_path = f"data_p{patient_number}/P{patient_number}_S{session}/anonymized"
            if(track_progress):
                print(data_path)
        else:
            data_path = patient_path+f"/P{patient_number}_S{session}/anonymized"
            if(track_progress):
                print(data_path)
        
        marker_info_session = get_markers_of_session(data_path, track_progress)
        
        total_filenames.append(marker_info_session[0])
        total_marker_names.update(marker_info_session[1])
        total_marker_uniques.append(marker_info_session[2])

    total_marker_uniques_flattened =  [uniques for session in total_marker_uniques for uniques in session]   
    patient_marker_info = total_filenames, total_marker_names, np.unique(total_marker_uniques_flattened)

    if print_result:
        _print_marker_information(patient_number, last_session_number, patient_marker_info)

    return (patient_marker_info)    

def _print_marker_information(patient_nr, last_session, patient_marker_info):
    print(f"Patient {patient_nr} marker information of all sessions 1-{last_session}")

    print("-------------------- Unique markers --------------------")
    print(patient_marker_info[2]) # uniques

    print("-------------------- Marker information --------------------")
    for i in patient_marker_info[1].keys():
        print(f"{i}:{patient_marker_info[1].get(i)}")

    print("-------------------- Loaded filenames --------------------")
    for session in patient_marker_info[0]:
        for filename in session:  
            print(filename)  


def get_session_filenames_with_markers(data_path, marker_list=None, track_progress=True):
    """
    Get all filenames of a session that have the given markers. If marker_list is None, return all unique markers per filename.
    
    Parameters:
    - data_path (str): path to data
    - marker_list (list | None): list of markers to include in the return. if None, include all unique markers for each filename.
    - track_progress (Boolean): True if you want to follow the function through printing the progress. if False, nothing is printed.

    Output:
    - filenames (dict): dictionary with filenames as keys and, as value to a key, a list of (marker, count) tuples for each marker

    Example usage:
    > session_path = "data_p1/P1_S3/anonymized"
    > filenames_with_marker = get_session_filenames_with_markers(session_path) 
    """

    data_dir = Path.cwd() / data_path
    header_files = data_dir.glob("auditoryAphasia*.vhdr")
    filenames = dict()
    marker_uniques = list()

    for f in header_files:
        #raw_data = _load_and_preprocess_raw(f)
        raw_data = mne.io.read_raw_brainvision(f, misc=non_eeg_channels, preload=True)

        # To store metadata 
        parent1 = f.parent.name            # immediate parent ("anonymized")
        parent2 = f.parent.parent.name     # grandparent (e.g. "P1_S5")
        filename = f"{parent2}/{parent1}/{f.name}"

        evs = mne.events_from_annotations(raw_data)
        all_markers = evs[0][:,2]
        marker_uniques, marker_counts = np.unique(all_markers, return_counts=True)
        filename_list = list()

        for m, mrk in enumerate(marker_uniques):
            if marker_list is None:
                filename_list.append((mrk, marker_counts[m]))
            elif mrk in marker_list:
                filename_list.append((mrk, marker_counts[m]))

        filenames.update({filename: filename_list})    
        
        if(track_progress):
            print(f"Loaded {parent2}/{parent1}/{f.name}")

    return filenames
        
def all_runs_contain_marker(marker_list, filenames = None, data_path = None):
    """
    Returns whether all runs contain the markers in a given marker_list
    
    Example usage:
    > filenames_p1_s1 = get_session_filenames_with_markers("data_p1/P1_S1/anonymized")
    > print(all_runs_contain_marker(filenames=filenames_p1_s1, marker_list=[201, 207]))
    """

    if filenames is None:
        if data_path is None:
            raise ValueError("Both filenames and data_path are not defined... Cannot find a file to read markers from.")
        filenames = get_session_filenames_with_markers(data_path, track_progress=False)

    for filename in filenames: # for each filename (i.e., each run)
        for marker in marker_list:
            marker_found = False
            for tuple in filenames.get(filename): # for each unique marker in the filename
                if marker == tuple[0]:
                    marker_found = True
            if not marker_found:
                return False
            
    return True    
