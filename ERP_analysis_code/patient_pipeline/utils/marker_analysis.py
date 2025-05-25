"""
Script to process and extract marker information from the auditory aphasia patient data.

Instructions:
- Fill in `data_path`, `patient_number`, `last_session_number`
- Important note: the functions work for data that is stored in the following way: ".../P01a/P1_S1/anonymized" which contains all .vhdr files for session 1 of patient 1. 
  The functions access the session content through: data_path + f"/P{patient_number}_S{session}/anonymized"

"""

from marker_functions import log_patient_marker_info, log_patient_odd_markers
import mne
import warnings
mne.set_log_level('WARNING')
warnings.filterwarnings("ignore") 

# Please fill in the following variables -----------------------------------------------------------------------------
data_path = "path/to/patient" 
data_path = "B:/anonymized_data/P04a"
patient_number = 4
last_session_number = 20

# Run script ---------------------------------------------------------------------------------------------------------
log_patient_marker_info(patient_number, last_session_number, data_path, f"p{patient_number}_marker_information.log")
log_patient_odd_markers(patient_number, last_session_number, data_path, f"p{patient_number}_odd_markers.log")

print("Done")
