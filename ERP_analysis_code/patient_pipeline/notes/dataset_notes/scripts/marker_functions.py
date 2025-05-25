"""
Module with functions to process and extract marker information from the auditory aphasia patient data.
Run the code through script `marker_analysis.py`
"""

import mne
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

def start_logging(log_file_name):
    logging.basicConfig(
        filename=log_file_name,
        filemode="w",
        level=logging.DEBUG,
        format='%(message)s')

def close_logging():
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)        

def log_patient_odd_markers(patient_nr, last_session, patient_path, log_name):
    """
    Creates a log file and write for each session, the common and odd markers across all runs. Also logs the filenames containing odd markers.

    Example usage:
    > log_patient_odd_markers(patient_nr = 1, last_session = 18, log_name = f"p1_odd_markers.log")
    """

    if log_name is None:
        log_name = f"p{patient_nr}_odd_markers.log"
    if not isinstance(patient_path, str):
        raise TypeError("patient_path must be a string")
    log_name = f"p{patient_nr}_odd_markers.log"

    start_logging(log_name)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp}")
    logging.info(f"Patient {patient_nr} marker information of all sessions 1-{last_session}")  

    string1 = "\nCommon markers across all runs, for each session"
    string2 = "\nMarkers that do not appear in all runs, for each session"
    odd_markers = []

    for session in range(1,last_session+1):
        try:  
            data_path = patient_path+f"/P{patient_nr}_S{session}/anonymized"
        except ValueError as e:
            print("Error while loading the session: ", e)  

        x = (get_markers_of_session(data_path))
        string1 += f"\nSession {session}: {(x[3])}"

        odd_markers.append(x[4])
        string2 += (f'\nSession {session}: {("-" if x[4]==set() else x[4])}')

    logging.info(string1)  
    logging.info(string2)

    odd_markers = np.unique([m for mrks in odd_markers for m in mrks])

    logging.info(f"\nSearching for the uncommon markers: {odd_markers}")
    logging.info("[filename]: [(marker, count)]")
    print("Searching through all runs for odd markers... (this may take up to 1 min)")

    for session in range(1,last_session+1):
        try:  
            data_path = patient_path+f"/P{patient_nr}_S{session}/anonymized"
        except ValueError as e:
            print("Error while loading the session: ", e)  
        
        filenames = get_session_filenames_with_markers(data_path, marker_list= odd_markers)
        for f in filenames:
            if len(filenames.get(f))>0:
                logging.info(f"{f}: {filenames.get(f)}")

    logging.info("\nEnd of log file")
    close_logging()

def log_patient_marker_info(patient_nr, last_session_nr, patient_path, log_name = None):
    """
    Creates a log file for a patient with all unique markers, all marker types, and all filenames. 
    Result is stored in p*_marker_information.log by Default.

    Parameters:
    - patient_nr (int)
    - last_session_nr (int)
    - patient_path (string): path to patient data. Note that this code only works if the path to a session is e.g. patient_path/P1_S1/anonymized for patient 1, session 1.
    - log_name (string): name of log file to write output to.

    Example usage:
    > patient = 1
    > last_session = 18
    > patient_path = f"B:/anonymized_data/P0{patient}a"
    > patient_marker_info = log_patient_marker_info(patient, last_session, patient_path = patient_path)
        
    """
    if log_name is None:
        log_name = f"p{patient_nr}_marker_information.log"
    if not isinstance(patient_path, str):
        raise TypeError("patient_path must be a string")

    total_marker_uniques = list()
    total_marker_names = dict()
    total_filenames = list()

    for session in range(1,last_session_nr+1):
        try:  
            data_path = patient_path+f"/P{patient_nr}_S{session}/anonymized"
        except ValueError as e:
            print("Error while loading the session: ", e)        

        marker_info_session = get_markers_of_session(data_path) # obtain tuple with marker info

        total_marker_uniques.append(marker_info_session[0])
        total_marker_names.update(marker_info_session[1])
        total_filenames.append(marker_info_session[2])

    total_marker_uniques_flattened =  [uniques for session in total_marker_uniques for uniques in session]   
    patient_marker_info = total_filenames, total_marker_names, np.unique(total_marker_uniques_flattened)

    _log_marker_information(patient_nr, last_session_nr, patient_marker_info, log_name=log_name)

def _log_marker_information(patient_nr, last_session_nr, patient_marker_info, log_name):
    start_logging(log_name)
    logging.info(f"Patient {patient_nr} marker information of all sessions 1-{last_session_nr}")
    logging.info("-------------------- Unique markers --------------------")
    logging.info(patient_marker_info[2])
    logging.info("-------------------- Marker information --------------------")
    for i in patient_marker_info[1].keys():
        logging.info(f"{i}:{patient_marker_info[1].get(i)}")
    logging.info("-------------------- Loaded filenames --------------------")
    for session in patient_marker_info[0]:
        for filename in session:  
            logging.info(filename)  

    close_logging()

def get_session_filenames_with_markers(session_path, marker_list=None):
    """
    Get all filenames of a session that have the given markers. If marker_list is None, return all unique markers per filename.
    
    Parameters:
    - session_path (str): path to session with .vhdr files
    - marker_list (list | None): list of markers to include in the output. if None, include all unique markers for each filename.

    Output:
    - filenames (dict): dictionary with filenames as keys and, as value to a key, a list of (marker, count) tuples for each marker

    Example usage:
    > session_path = "data_p1/P1_S3/anonymized"
    > filenames_with_marker = get_session_filenames_with_markers(session_path) 
    """

    data_dir = Path.cwd() / session_path
    header_files = data_dir.glob("auditoryAphasia*.vhdr")
    filenames = dict()
    marker_uniques = list()

    for f in header_files:
        raw_data = mne.io.read_raw_brainvision(f)

        # To store metadata 
        parent1 = f.parent.name            # immediate parent ("anonymized")
        parent2 = f.parent.parent.name     # grandparent (e.g. "P1_S1")
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

    return filenames
        
def get_markers_of_session(data_path):
    """
    Get all unique markers of a session, all marker names, all filenames, all common markers among all runs, all odd markers among all runs

    The output is a tuple and will be used in log_patient_marker_info() and log_patient_odd_markers()
    
    Output:
        (
        all_marker_uniques, 
        marker_names,
        filenames,
        sorted(common_markers), 
        odd_markers
        )
    """

    data_dir = Path.cwd() / data_path
    print(f"Extracting markers from {data_dir}...")
    header_files = data_dir.glob("auditoryAphasia*.vhdr")
    if next(header_files, None) is None:
        raise Exception("The given directory contains no auditoryAphasia*.vhdr files")

    all_marker_uniques = []
    marker_names = dict() 
    filenames = []
    common_markers = None

    for f in header_files:
        raw_data = mne.io.read_raw_brainvision(f)

        # To store metadata 
        parent1 = f.parent.name            # immediate parent ("anonymized")
        parent2 = f.parent.parent.name     # grandparent (e.g. "P1_S5")
        filenames.append(f"{parent2}/{parent1}/{f.name}")

        # Extract unique markers and marker names
        evs = mne.events_from_annotations(raw_data)
        marker_names.update(evs[1])
        run_marker_uniques = set(evs[0][:,2])
        all_marker_uniques.append(run_marker_uniques)

        # Store common and odd markers
        if common_markers is None:
            common_markers = run_marker_uniques
        else:
            common = common_markers & run_marker_uniques
            common_markers = common

    all_marker_uniques = np.unique([mrks for run in all_marker_uniques for mrks in run])
    odd_markers = set(all_marker_uniques) - (common_markers)

    return (
        all_marker_uniques, 
        marker_names, 
        filenames,
        sorted(common_markers), 
        odd_markers
        )