
import numpy as np
import matplotlib.pyplot as plt
from db import patients_db
import pickle

# Epoch-wise performance 

def extract_ews_v3(performances, patient_nr: int, last_online_session: int, strategy="", verbose=False, version_suffix="", clf="btlda"):
    """Extract epoch-wise accuracy scores from pickle files. Return scores in np array"""

    if verbose:
        print(f"Extracting from performances with the following keys: \n{performances.keys()}")
        print(f"strategy: {strategy}")

    #ews_transfer_lda = np.zeros(last_online_session-2)
    

    session = 3 # first online session
    p = patient_nr
    extra = 0

    if p == 8:
        session = 4
        extra +=1
    
    ews_btlda = np.zeros(last_online_session-2-extra)

    for i in range(last_online_session-2-extra):
        ews_btlda[i] = performances.get(f'p{p}_{strategy}_s{session}{version_suffix}').get('epoch-wise').get(clf)
        session+=1
    return ( 
        ews_btlda)  

# Trial-wise performance

def extract_tws_v3(performances, patient_nr: int, last_online_session: int, strategy="", verbose=False, version_suffix="", clf="btlda"):
    """Extract trial-wise accuracy scores from pickle files. Return scores in np array"""

    if verbose:
        print(f"Extracting from performances with the following keys: \n{performances.keys()}")
        print(f"strategy: {strategy}")

    session = 3 # first online session
    p = patient_nr
    extra = 0

    if p==8:
        session = 4
        extra += 1

    tws_btlda = np.zeros(last_online_session-2-extra)

    for i in range(last_online_session-2-extra):
        tws_btlda[i] = performances.get(f'p{p}_{strategy}_s{session}{version_suffix}').get('trial-wise').get(clf)
        session+=1
    return (
        tws_btlda)  



def grand_average(all_patients_dictionary):
    """Compute the grand average score (across all patients) for every session. Return scores in np array"""

    clf_all_scores = []
    for idx,p in enumerate(all_patients_dictionary):
        clf_all_scores.append(all_patients_dictionary.get(f'p{idx+1}'))

    sessions = np.zeros((25))
    for s,session in enumerate(sessions):
        session_average = []
        for e,element in enumerate(clf_all_scores):
            if e == 7 and s==0: # skip the first session of patient 8
                continue
            if e==7:
                if len(element) > s:
                    session_average.append(element[s-1])
            else:
                if len(element) > s:
                    session_average.append(element[s])

        #print("averaging across patients/elements gives: ", np.mean(session_average))            
        sessions[s] = np.mean(session_average)  
    return sessions   

def plot_all_patients(acc_all_patients, grand_avg = None, title="(empty)", ylabel="(empty)", ylim=(0.4,1)):
    colors=['lightblue','lightblue','blue','violet','lightgreen','green','tomato', 'darkblue', 'grey', 'gold', 'rosybrown']
    plt.figure(figsize=(18,10))
    for i in range (1,11):
        color = colors[i]
        y = acc_all_patients.get(f'p{i}')
        if i!=8:
            plt.plot(np.arange(0,len(y)), y, label=f"patient {i}", color=color)
        else:
            plt.plot(np.arange(1,len(y)), y[:-1], label=f"patient {i}", color=color)
    if grand_avg is not None:
        plt.plot(grand_avg, label="grand average", color = 'black', linewidth=2, linestyle='dashed')        
    plt.ylabel(ylabel)
    plt.xlabel('Session')
    plt.ylim(ylim)
    plt.xticks(np.arange(0,25), np.arange(3,28))    
    plt.legend()    
    plt.title(title)   
    plt.grid() 
    plt.show()


def get_scores_all_patients(pickle_suffix="", strategy="", classifier="", score=None, verbose=False, version_suffix=""):
    """
    Collects epoch-wise/trial-wise accuracy scores for all sessions, for every patient. Stores all scores together in a dictionary.

    Parameters:
    - pickle_suffix: e.g. "performances_v1"
    - strategy: e.g. "transfer" or "static" etc.
    - classifier: "btlda" or "slda"
    - verbose: True for printing the process, False otherwise
    
    Output:
    - ews_clf_all_combined (dict):{
                                    "p1": [score1, score2, ...],
                                    "p2": [score1, score2, ...],
                                    ...
                                    "p10": [score1, score2, ...]
                                    }

    Example usage:
    > # Collecting ews scores of Transfer Fixed BT-LDA
    > strategy = "transfer"
    > clf = "btlda"
    > pickle_name = "performances_v1"
    > ews_btlda_all_results = ews_all_patients(pickle_name, strategy, clf, verbose=False)                                
    """
    
    # To do: add check for right strategy values (& check for pickle suffix)

    assert classifier in ["btlda", "slda"], "Given classifier is not recognized. Accepts only btlda or slda"
    assert score in ["ews", "tws"], "Invalid score given. The following scores are valid input: ['ews', 'tws']"

    # Combine ews for all patients
    scores_all_patients = dict()

    for id in patients_db:

        info = patients_db.get(id)
        patient = info.get('patient_nr')
        last_session = info.get('last_session')
        calibration_selection = info.get('selection')

        if verbose:
            print("patient: ", patient)
            print("last session", last_session)
            print("calibration_selection", calibration_selection)

        with open(f'p{patient}_{pickle_suffix}.pkl', 'rb') as f:
            performances_new = pickle.load(f)    

        if score=="ews":   
            scores_transfer_btlda = extract_ews_v3(performances_new, patient, last_session-1, strategy, verbose=verbose, version_suffix=version_suffix, clf=classifier) 
        elif score=="tws":
            scores_transfer_btlda = extract_tws_v3(performances_new, patient, last_session-1, strategy, verbose, version_suffix=version_suffix, clf=classifier)

        scores_all_patients.update({f"p{patient}": (scores_transfer_btlda)})

    return scores_all_patients
