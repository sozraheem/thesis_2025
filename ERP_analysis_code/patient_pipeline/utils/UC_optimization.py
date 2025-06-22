from itertools import product
import numpy as np
import os
from utils.db import patients_db
import pickle
from utils.run_patient_v3 import run_patient_online_sessions_CC_UC_pairs

def create_version_dictionary(UC_pairs):
    """Creates a dictionary with version numbers mapped to UC-pairs"""
    UC_dict = dict()
    for v,UC_pair in enumerate(UC_pairs):
        UC_m = UC_pair[0]
        UC_cov = UC_pair[1]
        UC_dict.update({v:{
                            "UC_mean": UC_m,
                            "UC_cov": UC_cov
                            }
                        })

    for UC in UC_dict.keys():
        print(f"v{UC} - UC_mean: {UC_dict.get(UC).get("UC_mean")} - UC_cov {UC_dict.get(UC).get("UC_cov")}")
    
    return UC_dict


def run_uc_grid_search():
    """Run UC grid search on a group of patients (either odds or evens)"""

    ### REDUCED RANGE
    UC_mean_exponents = np.arange(-7, -2)  
    UC_mean_range = 0.5 * (2.0 ** UC_mean_exponents)

    UC_cov_exponents = np.arange(-17, 2)  
    UC_cov_range = 0.5 * (2.0 ** UC_cov_exponents)

    UC_pairs = list(product(UC_mean_range, UC_cov_range))
    print(UC_pairs)

    # Store versions with corresponding UC-pair
    UC_dict = create_version_dictionary(UC_pairs=UC_pairs)
    pkl_dir = f"results_UC"
    os.makedirs(pkl_dir, exist_ok=True)
    with open(os.path.join(pkl_dir, "versions.pkl"), 'wb') as f:
        pickle.dump(UC_dict, f)   

    # Perform grid search over UC-pairs
    for v,UC_pair in enumerate(UC_pairs):
        UC_m = UC_pair[0]
        UC_cov = UC_pair[1]
        print(f"v: {v} - UC_m: {UC_m} - UC_cov: {UC_cov}")
        
        # Run CC on all even patients
        if v>-1:
            for id in patients_db:
                if id%2==0: 
                    info = patients_db.get(id)
                    patient = info.get('patient_nr')
                    last_session = info.get('last_session')
                    calibration_selection = info.get('selection')

                    print("patient: ", patient)
                    print("last session", last_session)
                    print("calibration_selection", calibration_selection)

                    performances = run_patient_online_sessions_CC_UC_pairs(patient=patient, last_session_nr=last_session, calibration_selection=calibration_selection, UC_mean=UC_m, UC_cov=UC_cov, version=v)
                    # Save pickle
                    pkl_dir = f"results_UC/v{v}/pickles"
                    os.makedirs(pkl_dir, exist_ok=True)
                    with open(os.path.join(pkl_dir, f"p{patient}_cc_v{v}.pkl"), 'wb') as f:
                        pickle.dump(performances, f)


    
        

