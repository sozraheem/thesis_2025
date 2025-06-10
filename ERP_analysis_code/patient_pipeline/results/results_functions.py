
import numpy as np
import matplotlib.pyplot as plt


### Epoch-wise accuracy

def extract_ews(performances, patient_nr: int, last_online_session: int, strategy=""):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    print(f"strategy: {strategy}")

    #ews_transfer_lda = np.zeros(last_online_session-2)
    ews_slda = np.zeros(last_online_session-2)
    ews_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr
    extra = 0

    if p == 8:
        session = 4
        extra +=1
    
    for i in range(last_online_session-2-extra):
        ews_slda[i] = performances.get(f'p{p}_{strategy}_s{session}').get('epoch-wise').get('slda')
        ews_btlda[i] = performances.get(f'p{p}_{strategy}_s{session}').get('epoch-wise').get('btlda')
        session+=1
    return ( 
        ews_slda, 
        ews_btlda)  

def extract_tws(performances, patient_nr: int, last_online_session: int, strategy=""):
    print(f"Extracting from performances with the following keys: \n{performances.keys()}")
    print(f"strategy: {strategy}")

    tws_slda = np.zeros(last_online_session-2)
    tws_btlda = np.zeros(last_online_session-2)

    session = 3 # first online session
    p = patient_nr

    if p==8:
        session = 4

    # if p==8: session = 4 (for patient 8)
    
    for i in range(last_online_session-2):
        tws_slda[i] = performances.get(f'p{p}_{strategy}_s{session}').get('trial-wise').get('slda')
        tws_btlda[i] = performances.get(f'p{p}_{strategy}_s{session}').get('trial-wise').get('btlda')
        session+=1
    return ( 
        tws_slda, 
        tws_btlda)  

def grand_average(ews_clf_all_combined):
    sessions = np.zeros((25))
    for s,session in enumerate(sessions):
        session_average = []
        print("session: ",s)
        for e,element in enumerate(ews_clf_all_combined):
            if e == 7 and s==0:
                continue
            if e==7:
                if len(element) > s:
                    print(f"adding element (patient {e+1}): {element[s-1]}")
                    session_average.append(element[s-1])
            else:
                if len(element) > s:
                    print(f"adding element (patient {e+1}): {element[s]}")
                    session_average.append(element[s])

        print("averaging elements gives: ", np.mean(session_average))            
        sessions[s] = np.mean(session_average)  
    return sessions   

def plot_all_patients(acc_all_patients, grand_avg = None):
    colors=['lightblue','lightblue','blue','violet','lightgreen','green','tomato', 'darkblue', 'grey', 'gold', 'rosybrown']
    plt.figure(figsize=(18,10))
    for i in range (1,11):
        color = colors[i]
        y = acc_all_patients.get(f'p{i}')[1]
        if i!=8:
            plt.plot(np.arange(0,len(y)), y, label=f"patient {i}", color=color)
        else:
            plt.plot(np.arange(1,len(y)), y[:-1], label=f"patient {i}", color=color)
    if grand_avg is not None:
        plt.plot(grand_avg, label="grand average", color = 'black', linewidth=2, linestyle='dashed')        
    plt.ylabel('Epoch-wise accuracy')
    plt.xlabel('Session')
    plt.ylim(0.4,1)
    plt.xticks(np.arange(0,25), np.arange(3,28))    
    plt.legend()    
    plt.title('Transfer BT-LDA - epoch-wise accuracy')   
    plt.grid() 
    plt.show()


# def extract_ews_transfer(performances, patient_nr: int, last_online_session: int):
#     print(f"Extracting from performances with the following keys: \n{performances.keys()}")
#     ews_transfer_lda = np.zeros(last_online_session-2)
#     ews_transfer_slda = np.zeros(last_online_session-2)
#     ews_transfer_btlda = np.zeros(last_online_session-2)

#     session = 3 # first online session
#     p = patient_nr

#     # if p==8: session = 4 (for patient 8)
    
#     for i in range(last_online_session-2):
#         if session==3:
#             ews_transfer_lda[i] = performances.get(f'p{p}_transfer_s{session}').get('epoch-wise').get('lda')
#             ews_transfer_slda[i] = performances.get(f'p{p}_transfer_s{session}').get('epoch-wise').get('slda')
#             ews_transfer_btlda[i] = performances.get(f'p{p}_transfer_s{session}').get('epoch-wise').get('btlda')
#         else:
#             ews_transfer_lda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('epoch-wise').get('lda')
#             ews_transfer_slda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('epoch-wise').get('slda')
#             ews_transfer_btlda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('epoch-wise').get('btlda')
#         session+=1
#     return (ews_transfer_lda, ews_transfer_slda, ews_transfer_btlda)       

# def extract_ews_window(performances, patient_nr: int, last_online_session: int):
#     print(f"Extracting from performances with the following keys: \n{performances.keys()}")
#     ews_window_lda = np.zeros(last_online_session-2)
#     ews_window_slda = np.zeros(last_online_session-2)
#     ews_window_btlda = np.zeros(last_online_session-2)

#     session = 3 # first online session
#     p = patient_nr

#     # if p==8: session = 4 (for patient 8)
    
#     for i in range(last_online_session-2):
#         ews_window_lda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('epoch-wise').get('lda')
#         ews_window_slda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('epoch-wise').get('slda')
#         ews_window_btlda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('epoch-wise').get('btlda')
#         session+=1

#     return (ews_window_lda, ews_window_slda, ews_window_btlda)       

# def extract_ews_static(performances, patient_nr: int, last_online_session: int):
#     print(f"Extracting from performances with the following keys: \n{performances.keys()}")
#     ews_static_lda = np.zeros(last_online_session-2)
#     ews_static_slda = np.zeros(last_online_session-2)
#     ews_static_btlda = np.zeros(last_online_session-2)

#     session = 3 # first online session
#     p = patient_nr

#     # if p==8: session = 4 (for patient 8)
    
#     for i in range(last_online_session-2):
#         ews_static_lda[i] = performances.get(f'p{p}_static_s{session}').get('epoch-wise').get('lda')
#         ews_static_slda[i] = performances.get(f'p{p}_static_s{session}').get('epoch-wise').get('slda')
#         ews_static_btlda[i] = performances.get(f'p{p}_static_s{session}').get('epoch-wise').get('btlda')
#         session+=1

#     return (ews_static_lda, ews_static_slda, ews_static_btlda)   

# ### Trial-wise accuracy

# def extract_tws_transfer(performances, patient_nr: int, last_online_session: int):
#     print(f"Extracting from performances with the following keys: \n{performances.keys()}")
#     tws_transfer_lda = np.zeros(last_online_session-2)
#     tws_transfer_slda = np.zeros(last_online_session-2)
#     tws_transfer_btlda = np.zeros(last_online_session-2)

#     session = 3 # first online session
#     p = patient_nr

#     # if p==8: session = 4 (for patient 8)
    
#     for i in range(last_online_session-2):
#         if session==3:
#             tws_transfer_lda[i] = performances.get(f'p{p}_transfer_s{session}').get('trial-wise').get('lda')
#             tws_transfer_slda[i] = performances.get(f'p{p}_transfer_s{session}').get('trial-wise').get('slda')
#             tws_transfer_btlda[i] = performances.get(f'p{p}_transfer_s{session}').get('trial-wise').get('btlda')
#         else:
#             tws_transfer_lda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('trial-wise').get('lda')
#             tws_transfer_slda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('trial-wise').get('slda')
#             tws_transfer_btlda[i] = performances.get(f'p{p}_transfer_fixed_s{session}').get('trial-wise').get('btlda')
#         session+=1
#     return (tws_transfer_lda, tws_transfer_slda, tws_transfer_btlda)  

# def extract_tws_window(performances, patient_nr: int, last_online_session: int):
#     print(f"Extracting from performances with the following keys: \n{performances.keys()}")
#     tws_window_lda = np.zeros(last_online_session-2)
#     tws_window_slda = np.zeros(last_online_session-2)
#     tws_window_btlda = np.zeros(last_online_session-2)

#     session = 3 # first online session
#     p = patient_nr

#     # if p==8: session = 4 (for patient 8)
    
#     for i in range(last_online_session-2):
#         tws_window_lda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('trial-wise').get('lda')
#         tws_window_slda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('trial-wise').get('slda')
#         tws_window_btlda[i] = performances.get(f'p{p}_adaptive_sw_s{session}').get('trial-wise').get('btlda')
#         session+=1

#     return (tws_window_lda, tws_window_slda, tws_window_btlda)       


# def extract_tws_static(performances, patient_nr: int, last_online_session: int):
#     print(f"Extracting from performances with the following keys: \n{performances.keys()}")
#     tws_static_lda = np.zeros(last_online_session-2)
#     tws_static_slda = np.zeros(last_online_session-2)
#     tws_static_btlda = np.zeros(last_online_session-2)

#     session = 3 # first online session
#     p = patient_nr

#     # if p==8: session = 4 (for patient 8)
    
#     for i in range(last_online_session-2):
#         tws_static_lda[i] = performances.get(f'p{p}_static_s{session}').get('trial-wise').get('lda')
#         tws_static_slda[i] = performances.get(f'p{p}_static_s{session}').get('trial-wise').get('slda')
#         tws_static_btlda[i] = performances.get(f'p{p}_static_s{session}').get('trial-wise').get('btlda')
#         session+=1

#     return (tws_static_lda, tws_static_slda, tws_static_btlda)   