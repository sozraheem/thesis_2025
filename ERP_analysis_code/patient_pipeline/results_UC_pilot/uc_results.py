
# Functions to extract and store results from the grid search on UC-pairs

import pandas as pd
import pickle
import numpy as np
from patient_pipeline.results.results_functions import extract_ews_v3, extract_tws_v3
from patient_pipeline.results.db import patients_db

### Function to call ---------------------------------------------------------------------------------------------------

def df_UC_results_patient(patient, max_version=76, only_mean=False):
    """Return dataframe of UC-pair grid search - results are of a single patient
    
    Params:
    - patient (int): patient nr
    - max_version (int): last element in the grid search to include in results
    - only_mean (boolean): if True, store only the average across sessions. if False, store separate scores per session.

    Output:
    - df: pandas dataframe with results
    """

    # extract patient info from database db
    info = patients_db.get(patient)
    lon = info.get('last_session')-1
    
    # read scores of patient
    data_dict = _create_dict_UC_results_patient(patient=patient, last_online=lon, max_version=max_version, only_mean=only_mean)
    df = _create_dataframe(data_dict)
    return df
    #df.to_csv(f"all_UC_results_p{patient}.csv")

def df_UC_results_combined(group=None):
    """Return dataframe of UC-pair grid search - results are of a group of patients (even/odd)"""

    if group not in ["even", "odd"]:
        raise ValueError("Invalid parameter given for group")

    data_dict = _create_dict_UC_results_combined(group=group)
    df = _create_dataframe(data_dict)
    return df
    #df.to_csv(f"mean_UC_results_{group}.csv")    

### Helper functions ---------------------------------------------------------------------------------------------------

def _create_dict_UC_results_combined(group=None, max_version=76):
    """
    Extract for a a group of patients, all (or averaged) results of grid search on UC-pairs and return a dictionary
    
    Params:
    - patient (int): patient nr
    - last_online (int): last online session of that patient
    - max_version (int): last element in the grid search to include in results
    - only_mean (boolean): if True, store only the average across sessions. if False, store separate scores per session.

    Output:
    - dictionary (to create pandas dataframe from)
    """

    with open(f'versions.pkl', 'rb') as f:
        versions_dict = pickle.load(f) 
    
    # Check if only even or odd patients should be considered
    if group not in ["even", "odd"]:
        raise ValueError("Invalid parameter given for group")
    
    elif group == "even":
        even = True
    else:
        even = False


    UC_pairs = []
    data = []

    for v,version in enumerate(versions_dict.keys()):
        ews_scores = []
        tws_scores = []
        if v<max_version:
            UC_m = versions_dict.get(version).get("UC_mean")
            UC_c = versions_dict.get(version).get("UC_cov")

            for id in patients_db:
                
                # Consider only even patients
                if even and id%2==0:
                    info = patients_db.get(id)
                    p = info.get('patient_nr')
                    lon = info.get('last_session')-1

                    with open(f'v{v}/pickles/p{p}_cc_v{v}.pkl', 'rb') as f:
                        p_version_scores = pickle.load(f) 
                    
                    # get epoch-wise (ews) and trial-wise (tws) scores
                    ews_cc= extract_ews_v3(performances= p_version_scores, patient_nr=p, last_online_session=lon, strategy="cc")
                    tws_cc = extract_tws_v3(performances= p_version_scores, patient_nr=p, last_online_session=lon, strategy="cc")
                    ews_scores.append(ews_cc.tolist())
                    tws_scores.append(tws_cc.tolist())

                # Consider only odd patients
                elif even is False and id%2 != 0:    
                    info = patients_db.get(id)
                    p = info.get('patient_nr')
                    lon = info.get('last_session')-1

                    with open(f'v{v}/pickles/p{p}_cc_v{v}.pkl', 'rb') as f:
                        p_version_scores = pickle.load(f) 
                    
                    ews_cc= extract_ews_v3(performances= p_version_scores, patient_nr=p, last_online_session=lon, strategy="cc")
                    tws_cc = extract_tws_v3(performances= p_version_scores, patient_nr=p, last_online_session=lon, strategy="cc")
                    ews_scores.append(ews_cc.tolist())
                    tws_scores.append(tws_cc.tolist())

            ews_scores = [x for sublist in ews_scores for x in sublist]
            tws_scores = [x for sublist in tws_scores for x in sublist]

            data.append({
                "v": v,
                "UC_mean": UC_m,
                "UC_cov": UC_c,
                "EWS_mean": np.array(ews_scores[:]).mean(),
                "TWS_mean": np.array(tws_scores[:]).mean()
            })

    return data

def _create_dict_UC_results_patient(patient, last_online, max_version=76, only_mean=False):
    """
    Extract for a single patient, all (or averaged) results of grid search on UC-pairs and return a dictionary
    
    Params:
    - patient (int): patient nr
    - last_online (int): last online session of that patient
    - max_version (int): last element in the grid search to include in results
    - only_mean (boolean): if True, store only the average across sessions. if False, store separate scores per session.

    Output:
    - dictionary (to create pandas dataframe from)
    """

    p = patient
    lon = last_online

    # Get the corresponding UC-pair (UC_mean, UC_cov) per version (i.e., per element in the grid search)
    with open(f'versions.pkl', 'rb') as f:
        versions_dict = pickle.load(f) 

    data = []    

    for v,version in enumerate(versions_dict.keys()):
        ews_scores = []
        tws_scores = []
        if v<max_version: # only consider the selected range in the grid search
            UC_m = versions_dict.get(version).get("UC_mean")
            UC_c = versions_dict.get(version).get("UC_cov")

            # Extract the ews and tws scores of the patient of that version
            with open(f'v{v}/pickles/p{p}_cc_v{v}.pkl', 'rb') as f:
                p_version_scores = pickle.load(f) 

                ews_cc = extract_ews_v3(performances= p_version_scores, patient_nr=p, last_online_session=lon, strategy="cc")
                tws_cc = extract_tws_v3(performances= p_version_scores, patient_nr=p, last_online_session=lon, strategy="cc")
                ews_scores.append(ews_cc.tolist())
                tws_scores.append(tws_cc.tolist())

            ews_scores = [x for sublist in ews_scores for x in sublist]
            tws_scores = [x for sublist in tws_scores for x in sublist]

            # Store scores in dataframe
            if p == 8: # patient 8 is an exception for whom the online sessions start at 4
                online_session = 4
            else:    
                online_session = 3

            # If only the averages per UC-pair ("version") should be stored
            if only_mean:
                data.append({
                    "patient":p,
                    "v": v,
                    "UC_mean": UC_m,
                    "UC_cov": UC_c,
                    "EWS_mean": np.mean(np.array(ews_scores[:])),
                    "TWS_mean": np.array(tws_scores[:]).mean()
                })

            # Store not the average, but all scores (separate sessions) per UC-pair
            else:    
                for s,session in enumerate(ews_scores):
                    data.append({
                        "patient":p,
                        "v": v,
                        "UC_mean": UC_m,
                        "UC_cov": UC_c,
                        "session": online_session,
                        "EWS": np.array(ews_scores[s]),
                        "TWS": np.array(tws_scores[s])
                    })
                    online_session+=1    

    return data

def _create_dataframe(dict):
    df_long = pd.DataFrame(dict)
    return df_long