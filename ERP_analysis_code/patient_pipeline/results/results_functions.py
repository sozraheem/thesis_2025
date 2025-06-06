
import numpy as np


### Epoch-wise accuracy

def extract_ews_transfer(performances, patient_nr: int, last_online_session: int):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    ews_transfer_lda = np.zeros(last_online_session-2)
    ews_transfer_slda = np.zeros(last_online_session-2)
    ews_transfer_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        if session==3:
            ews_transfer_lda[i] = performances.get(f'p{p}_transfer_s{session}').get('epoch-wise').get('lda')
            ews_transfer_slda[i] = performances.get(f'p{p}_transfer_s{session}').get('epoch-wise').get('slda')
            ews_transfer_btlda[i] = performances.get(f'p{p}_transfer_s{session}').get('epoch-wise').get('btlda')
        else:
            ews_transfer_lda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('epoch-wise').get('lda')
            ews_transfer_slda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('epoch-wise').get('slda')
            ews_transfer_btlda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('epoch-wise').get('btlda')
        session+=1
    return (ews_transfer_lda, ews_transfer_slda, ews_transfer_btlda)       

def extract_ews_window(performances, patient_nr: int, last_online_session: int):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    ews_window_lda = np.zeros(last_online_session-2)
    ews_window_slda = np.zeros(last_online_session-2)
    ews_window_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        ews_window_lda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('epoch-wise').get('lda')
        ews_window_slda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('epoch-wise').get('slda')
        ews_window_btlda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('epoch-wise').get('btlda')
        session+=1

    return (ews_window_lda, ews_window_slda, ews_window_btlda)       

def extract_ews_static(performances, patient_nr: int, last_online_session: int):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    ews_static_lda = np.zeros(last_online_session-2)
    ews_static_slda = np.zeros(last_online_session-2)
    ews_static_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        ews_static_lda[i] = performances.get(f'p{p}_static_s{session}').get('epoch-wise').get('lda')
        ews_static_slda[i] = performances.get(f'p{p}_static_s{session}').get('epoch-wise').get('slda')
        ews_static_btlda[i] = performances.get(f'p{p}_static_s{session}').get('epoch-wise').get('btlda')
        session+=1

    return (ews_static_lda, ews_static_slda, ews_static_btlda)   

### Trial-wise accuracy

def extract_tws_transfer(performances, patient_nr: int, last_online_session: int):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    tws_transfer_lda = np.zeros(last_online_session-2)
    tws_transfer_slda = np.zeros(last_online_session-2)
    tws_transfer_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        if session==3:
            tws_transfer_lda[i] = performances.get(f'p{p}_transfer_s{session}').get('trial-wise').get('lda')
            tws_transfer_slda[i] = performances.get(f'p{p}_transfer_s{session}').get('trial-wise').get('slda')
            tws_transfer_btlda[i] = performances.get(f'p{p}_transfer_s{session}').get('trial-wise').get('btlda')
        else:
            tws_transfer_lda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('trial-wise').get('lda')
            tws_transfer_slda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('trial-wise').get('slda')
            tws_transfer_btlda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('trial-wise').get('btlda')
        session+=1
    return (tws_transfer_lda, tws_transfer_slda, tws_transfer_btlda)  

def extract_tws_window(performances, patient_nr: int, last_online_session: int):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    tws_window_lda = np.zeros(last_online_session-2)
    tws_window_slda = np.zeros(last_online_session-2)
    tws_window_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        tws_window_lda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('trial-wise').get('lda')
        tws_window_slda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('trial-wise').get('slda')
        tws_window_btlda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('trial-wise').get('btlda')
        session+=1

    return (tws_window_lda, tws_window_slda, tws_window_btlda)       


def extract_tws_static(performances, patient_nr: int, last_online_session: int):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    tws_static_lda = np.zeros(last_online_session-2)
    tws_static_slda = np.zeros(last_online_session-2)
    tws_static_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        tws_static_lda[i] = performances.get(f'p{p}_static_s{session}').get('trial-wise').get('lda')
        tws_static_slda[i] = performances.get(f'p{p}_static_s{session}').get('trial-wise').get('slda')
        tws_static_btlda[i] = performances.get(f'p{p}_static_s{session}').get('trial-wise').get('btlda')
        session+=1

    return (tws_static_lda, tws_static_slda, tws_static_btlda)   