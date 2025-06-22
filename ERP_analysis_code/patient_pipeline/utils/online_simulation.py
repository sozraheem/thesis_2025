# Functions that simulate a single session
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from toeplitzlda.classification import ToeplitzLDA
from utils.my_toeplitzlda import MyToeplitzLDA, ShrinkageLinearDiscriminantAnalysis
from utils.feature_extraction_v2 import get_jumping_means, epoch_vectorizer_channelprime, load_or_extract_markers
from utils.preprocessing import _have_same_preprocessing, get_n_epochs, get_iteration_structure
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import logging

# to hide INFO/DEBUG logs
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

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

def log_filenames(filenames):
    if filenames is None:
        log_text = "No filenames were given when the training data was passed"
    else:
        log_text = f"data filenames: {filenames}"
    return log_text        

def log_preprocessing(preprocessing_dictionary):
    if preprocessing_dictionary is None:
        text = "No preprocessing configurations were passed..."
    else:    
        text = "------------------------- Preprocessing configurations -------------------------"
        keys = preprocessing_dictionary.keys()
        for key in keys:
            value = preprocessing_dictionary.get(key)
            text += f"\n{key}: {value}"
        #text += "\n--------------------------------------------------------------------------------"
    return text

def log_feature_extraction(features=None, X_shape=None, y_shape=None):
    if features is None:
        text = "No feature information was passed..."
    else:    
        fe_info = features.get('fe_info')
        text = "------------------------- Feature extraction -------------------------"
        # text += f"\ntime_interval_boundaries: {ival_bounds}"
        # text += f"\ndata_is_channel_prime: {True}" # this is hard coded
        
        text += f"\ntime_interval_boundaries: {fe_info.get('time_ivals')}"
        text += f"\ndata_is_channel_prime: {fe_info.get('is_channel_prime')}"
        if X_shape and y_shape is not None:
            text += f"\nX.shape (epochs, features): {X_shape}"
            text += f"\ny.shape (epochs,): {y_shape}"
        text += "\n----------------------------------------------------------------------"
    return text

def log_clf_parameters(clf):
    clf = clf[0]
    text = "------------------------- Classifier parameters -------------------------"
    text += f"\nClass means: \n{clf.cl_mean}"
    # text += f"\nGlobal cov matrix: \n{clf.C_w}"
    # text += f"\nWeight vector w: {clf.w}"
    # text += f"\nBias b: {clf.b}"
    text += "\n----------------------------------------------------------------------"
    return text

def online_cc_simulation(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, UC_mean:float = None, UC_cov:float = None, clf=None, adaptive_slda=False):
    """
    Online simulation withs convex combination adaptation. Update the classifier only after a trial has finished.

    See online_simulation() for documentation.
    """

    online_trials = online_data.get("trials")
    ppon = online_data.get('preprocessing')
    fnon = online_data.get('filenames')

    if clf is None:
        # If no classifier is passed, train the clf based on the training data
        # Extract information from data
        raw_calibration_trials = raw_calibration_data.get('trials')
        print("All calibration trials: ",len(raw_calibration_trials))
        ppcal = raw_calibration_data.get('preprocessing')
        fncal = raw_calibration_data.get('filenames')

        if log_process is not None:
            start_logging(log_process)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"New log file - {timestamp}")
            logging.info(f"Convex combination adaptation with UC_mean = {UC_mean} and UC_cov = {UC_cov}")
            logging.info("================================ Calibration ================================")
            logging.info(f"Calibration {log_filenames(fncal)}")
            logging.info(log_preprocessing(ppcal))

        # Feature extraction
        ival_bounds = calibration_features.get('fe_info').get('time_ivals')
        X_train, y_train = epoch_vectorizer_channelprime(raw_calibration_trials=raw_calibration_trials, ival_bounds=ival_bounds)

        ### Calibration -----------------------------------------------------------------------

        ## BT-LDA
        nch = (raw_calibration_trials[0][0]).info["nchan"]
        btlda = make_pipeline(MyToeplitzLDA(n_channels=nch),)
        classifier = btlda

        if adaptive_slda:
            slda = make_pipeline(ShrinkageLinearDiscriminantAnalysis(n_channels=nch))
            classifier = slda

        classifier.fit(X_train,y_train)

    # For CC updating, the classifier of the previous session is taken into every new session
    else:
        classifier = clf

    # if no previous classifier was passed, the training data was used to initialize the current classifier
    if log_process and clf is None:
        logging.info(log_feature_extraction(calibration_features, X_train.shape, y_train.shape))
        logging.info(f"Full loaded n_calibration_trials: {len(raw_calibration_trials)}")
        logging.info(f"Full loaded n_calibration_epochs: {get_n_epochs(raw_calibration_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(raw_calibration_trials)}")
        logging.info("================================ Online ================================")
        logging.info(f"Online {log_filenames(fnon)}")
        if _have_same_preprocessing(ppcal, ppon):
            logging.info("Same preprocessing configurations as for the calibration data")
        else:
            logging.info(log_preprocessing(ppon))
        logging.info(log_feature_extraction(online_features))
        logging.info(f"n_online_trials: {len(online_trials)}")
        logging.info(f"n_online_epochs {get_n_epochs(online_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(online_trials)}")
        logging.info("Online simulation starts")
        logging.info(f"Number of online trials: {len(online_trials)}, which is {len(online_trials)/6} runs")

    # If a classifier is passed, no training data was used. So, there is no new information about preprocessing and feature extraction
    if log_process and clf is not None:
        start_logging(log_process)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"New log file - {timestamp}")
        logging.info(f"Convex combination adaptation with UC_mean = {UC_mean} and UC_cov = {UC_cov}")
        logging.info("================================ Calibration ================================")
        logging.info(f"No training data was used. Instead, the classifier of the previous session was passed for classifier initialization")
        #logging.info(log_clf_parameters(clf))
        logging.info("================================ Online ================================")
        logging.info(f"Online {log_filenames(fnon)}")
        logging.info(log_preprocessing(ppon))
        logging.info(log_feature_extraction(online_features))
        logging.info(f"n_online_trials: {len(online_trials)}")
        logging.info(f"n_online_epochs {get_n_epochs(online_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(online_trials)}")
        logging.info("Online simulation starts")
        logging.info(f"Number of online trials: {len(online_trials)}, which is {len(online_trials)/6} runs")

    ### Online simulation ------------------------------------------------------------------

    # Extract relevant data, labels and the played words
    online_trial_targets = np.array([trial[0]["Target"].events[:,2][0] % 10 for trial in online_trials]) # The target word per trial
    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]            
    online_labels = np.array(online_labels) 
    online_words = [(iteration.events[:,2]%10) for trial in online_trials for iteration in trial]
    online_words = np.array(online_words) 

    # compute distances to the decision boundary per epoch
    signed_distances = np.zeros(len(online_labels))

    # get marker info
    online_info = load_or_extract_markers(online_features.get('pickle_path'), online_trials=online_trials)
    markers2 = online_info.get('markers')

    epoch_count = 0 
    played_word_count = 0
    features = online_features.get('features')

    # word decision after a trial
    trial_predictions = np.zeros(online_trial_targets.shape)

    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            run_nr = math.trunc(t/6)+1
            logging.info(f"------------------ Run {run_nr} Trial {t%6+1}  (total trials: {t+1}/{len(online_trials)}) ------------------")
            if adaptive_slda:
                logging.info("{epoch} | {word_id} | {adaptive sLDA} ")    
            else:    
                logging.info("{epoch} | {word_id} | {BTLDA} ")

        stim_distances = np.zeros((len(trial),6))

        X_new_list = []
        y_new_list = []

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                new_x = features[epoch_count]

                # Compute signed distance of stimulus to decision boundary 
                s3 = classifier.decision_function(new_x).item()
                signed_distances[epoch_count] = s3 
                
                if log_process:
                    marker = markers2[epoch_count]
                    logging.info(f"{epoch_count} \t| {marker} \t| {s3}")

                # for word decision
                word_id = online_words[played_word_count,s] - 1 
                stim_distances[i,word_id] = s3 

                ### adaptation (store data in training set for the new clf; updating takes place at the end of a trial)
                x = new_x
                y = online_labels[epoch_count]
                X_new_list.append(x)
                y_new_list.append(y)

                # note that we did not update our classifier (yet)
                epoch_count+=1 

            
            played_word_count += 1

        # End of trial
        word_means = np.mean(stim_distances, axis=0)
        best_guess = np.argmax(word_means)
        trial_predictions[t] = best_guess + 1

        ### Adaptation: update our classifier after a trial has finished
        # convert to np arrays
        X_new = np.vstack(X_new_list)
        y_new = np.array(y_new_list)

        # Update classifier
        classifier[0].update_cc(X_new=X_new, y_new=y_new, UC_mean=UC_mean, UC_cov=UC_cov)

        if log_process:
            logging.info("------------------ End of trial ------------------")
            if adaptive_slda:
                logging.info("{real_word} | {adaptive_sLDA_prediction} ")
            else:
                logging.info("{real_word} | {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {}".format(online_trial_targets[t],best_guess+1))
            logging.info("Adaptation: this trial has been used to train a new classifier")
            logging.info("Updated the current classifier through a convex combination with the new classifier")

    fpr, tpr, thresholds = metrics.roc_curve(online_labels,signed_distances) 

    if log_process:
        logging.info("End of online simulation")
        logging.info("------------------ Epoch-wise performance ------------------")
        if adaptive_slda:
            logging.info(f"AUC-ROC Adaptive sLDA: {metrics.auc(fpr, tpr):0.5f}")
        else:    
            logging.info(f"AUC-ROC BT-LDA: {metrics.auc(fpr, tpr):0.5f}")

    # print("------------------ Word prediction performance (per trial) ------------------")
    # print(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):0.5f}")

    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        if adaptive_slda:
            logging.info(f"Accuracy Adaptive sLDA: {np.mean(trial_predictions == online_trial_targets):.5f} ({np.sum(trial_predictions == online_trial_targets)} correct out of {len(online_trial_targets)})")
        else:    
            logging.info(f"Accuracy BT-LDA: {np.mean(trial_predictions == online_trial_targets):.5f} ({np.sum(trial_predictions == online_trial_targets)} correct out of {len(online_trial_targets)})")
        logging.info("\n\n\n")
        logging.info(log_clf_parameters(classifier))

        close_logging()
        
    if adaptive_slda:
        performances = {
            "epoch-wise": {
            "slda": metrics.auc(fpr, tpr),
        },
        "trial-wise": {
            "slda": np.mean(trial_predictions == online_trial_targets),
        },
        "trial_predictions":{
            "slda": trial_predictions,
            "true": online_trial_targets
        }
        }
    
    else:
        performances = {
        "epoch-wise": {
            "btlda": metrics.auc(fpr, tpr),
        },
        "trial-wise": {
            "btlda": np.mean(trial_predictions == online_trial_targets),
        },
        "trial_predictions":{
            "btlda": trial_predictions,
            "true": online_trial_targets
        }
        }

    return (performances,classifier)

def online_transfer_simulation_v2(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, title_text = ""):
    """
    Online simulation of Transfer Fixed BT-LDA (with no within-session updating)

    BT-LDA is trained on the calibration trials. Then, for the online trials in the online simulation, it will predict the label of every played word (epoch) within a trial and decode the target word at the end of every online trial. At the end of the simulation the predictions will be compared against the real labels and real target words. An epoch-wise (label prediction) and trial-wise (target word prediction) will be logged and plotted.

    Parameters:
    - raw_calibration_data (dict): calibration data, obtained from load_session_chached or load_complete_session (utils.preprocessing)
    - online_data (dict): online simulation data, obtained from load_session_chached or load_complete_session (utils.preprocessing)
    - calibration_features (dict): extracted features, obtained from load_features_chached
    - online_features (dict): extracted features, obtained from load_features_chached
    - log_process (string | None): if a string is passed, save the log file to that name. If None (default), do not log the process.

    Output:
    - performances (dict)

    """

    # Extract information from data
    raw_calibration_trials = raw_calibration_data.get('trials')
    print("All calibration trials: ",len(raw_calibration_trials))
    print("That is {} epochs\n".format(get_n_epochs(raw_calibration_trials)))
    ppcal = raw_calibration_data.get('preprocessing')
    fncal = raw_calibration_data.get('filenames')

    online_trials = online_data.get("trials")
    ppon = online_data.get('preprocessing')
    fnon = online_data.get('filenames')

    if log_process is not None:
        start_logging(log_process)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"New log file - {timestamp}")
        logging.info("================================ Calibration ================================")
        logging.info(f"Calibration {log_filenames(fncal)}")
        logging.info(log_preprocessing(ppcal))
            
    # Feature extraction
    ival_bounds = calibration_features.get('fe_info').get('time_ivals')

    X, y = epoch_vectorizer_channelprime(raw_calibration_trials=raw_calibration_trials, ival_bounds=ival_bounds)

    ### Calibration -----------------------------------------------------------------------

    nch = (raw_calibration_trials[0][0]).info["nchan"]
    btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
    btlda.fit(X,y)


    if log_process:
        logging.info(log_feature_extraction(calibration_features, X.shape, y.shape))
        logging.info(f"n_calibration_trials: {len(raw_calibration_trials)}")
        logging.info(f"n_calibration_epochs: {get_n_epochs(raw_calibration_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(raw_calibration_trials)}")
        logging.info("Trained all three classifiers on the calibration data.")
        logging.info("================================ Online ================================")
        logging.info(f"Online {log_filenames(fnon)}")
        if _have_same_preprocessing(ppcal, ppon):
            logging.info("Same preprocessing configurations as for the calibration data")
        else:
            logging.info(log_preprocessing(ppon))
        logging.info(log_feature_extraction(online_features))
        logging.info(f"n_online_trials: {len(online_trials)}")
        logging.info(f"n_online_epochs {get_n_epochs(online_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(online_trials)}")
        logging.info("Online simulation starts")
        logging.info(f"Number of online trials: {len(online_trials)}, which is {len(online_trials)/6} runs")

    ### Online simulation ------------------------------------------------------------------
           
    # Extract relevant data, labels and the played words
    #load_online_info_chached(online_trials, online_features.get('pickle_path'))
    online_trial_targets = np.array([trial[0]["Target"].events[:,2][0] % 10 for trial in online_trials]) # target word per trial
    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]            
    online_labels = np.array(online_labels) # conversion to np array is maybe not even needed
    online_words = [(iteration.events[:,2]%10) for trial in online_trials for iteration in trial]
    online_words = np.array(online_words) # conversion to np array is maybe not even needed

    online_info = load_or_extract_markers(online_features.get('pickle_path'), online_trials=online_trials)
    markers2 = online_info.get('markers')

    # store distances to the decision boundary per epoch
    signed_distances_btlda = np.zeros(len(online_labels))

    epoch_count = 0 
    played_word_count = 0
    features = online_features.get('features')

    # word decision after a trial
    trial_predictions_btlda = np.zeros(online_trial_targets.shape)


    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            run_nr = math.trunc(t/6)+1
            logging.info(f"------------------ Run {run_nr} Trial {t%6+1}  (total trials: {t+1}/{len(online_trials)}) ------------------")
            logging.info("{epoch} | {word_id} | {BTLDA} ")

        stim_distances_btlda = np.zeros((len(trial),6))

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                x = features[epoch_count]

                s3 = btlda.decision_function(x).item()
                signed_distances_btlda[epoch_count] = s3 # Compute signed distance of stimulus to decision boundary
                
                if log_process:
                    marker = markers2[epoch_count]
                    logging.info(f"{epoch_count} \t| {marker} \t| {s3}")

                # for word decision
                word_id = online_words[played_word_count,s] - 1 # get the word id of the current epoch/stimulus s
                stim_distances_btlda[i,word_id] = s3 
                epoch_count+=1
            
            played_word_count += 1

        # End of trial
        means_btlda = np.mean(stim_distances_btlda, axis=0) # get the mean distance for each word in the trial

        best_guess_btlda = np.argmax(means_btlda) # predict the word

        trial_predictions_btlda[t] = best_guess_btlda + 1 # convert to correct word id

        if log_process:
            logging.info("------------------ End of trial ------------------")
            logging.info("{real_word} | {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {}".format(online_trial_targets[t],best_guess_btlda+1))

    print("------------------ Epoch-wise performance ------------------")

    fpr_btlda, tpr_btlda, thresholds = metrics.roc_curve(online_labels,signed_distances_btlda) 
    
    if log_process:
        logging.info("End of online simulation")
        logging.info("------------------ Epoch-wise performance ------------------")
        logging.info(f"AUC-ROC BT-LDA: {metrics.auc(fpr_btlda, tpr_btlda):0.5f}")
    
    print("------------------ Word prediction performance (per trial) ------------------")
    print(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):0.5f}")


    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        logging.info(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):.5f} ({np.sum(trial_predictions_btlda == online_trial_targets)} correct out of {len(online_trial_targets)})")

        close_logging()
        
    performances = {
        "epoch-wise": {
            "btlda": metrics.auc(fpr_btlda, tpr_btlda),
        },
        "trial-wise": {
            "btlda": np.mean(trial_predictions_btlda == online_trial_targets),
        },
        "trial_predictions":{
            "btlda": trial_predictions_btlda,
            "true": online_trial_targets
        }
    }
    
    return performances


def online_window_simulation_v5(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, title_text = "", original_window_size = 3600):
    """
    Online simulation with of Adaptive Window BT-LDA. For every epoch, add that epoch to the training set and remove the oldest epoch from the training set. Update the classifiers only after a trial has finished.

    See online_simulation() for documentation.
    """

    # Extract information from data
    raw_calibration_trials = raw_calibration_data.get('trials')
    print("All calibration trials: ",len(raw_calibration_trials))
    print("That is {} epochs\n".format(get_n_epochs(raw_calibration_trials)))
    print(f"Original window size: {original_window_size} epochs")
    # Check if the training data contains enough epochs. If not, reduce the window size to the size of the training data
    window_size = np.min([original_window_size,get_n_epochs(raw_calibration_trials)])
    print(f"For this session, the window size will be: {window_size}")
    ppcal = raw_calibration_data.get('preprocessing')
    fncal = raw_calibration_data.get('filenames')

    online_trials = online_data.get("trials")
    ppon = online_data.get('preprocessing')
    fnon = online_data.get('filenames')

    if log_process is not None:
        start_logging(log_process)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"New log file - {timestamp}")
        logging.info("================================ Calibration ================================")
        logging.info(f"Calibration {log_filenames(fncal)}")
        logging.info(log_preprocessing(ppcal))

    # Feature extraction
    ival_bounds = calibration_features.get('fe_info').get('time_ivals')
    X_train, y_train = epoch_vectorizer_channelprime(raw_calibration_trials=raw_calibration_trials, ival_bounds=ival_bounds)
    # Selecting only the most recent data; select all data points until the window size
    if len(y_train>window_size):
        X_train = X_train[-window_size:]
        y_train = y_train[-window_size:]

    ### Calibration -----------------------------------------------------------------------

    ## BT-LDA
    nch = (raw_calibration_trials[0][0]).info["nchan"]
    btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
    btlda.fit(X_train,y_train)


    if log_process:
        logging.info(log_feature_extraction(calibration_features, X_train.shape, y_train.shape))
        logging.info(f"Full loaded n_calibration_trials: {len(raw_calibration_trials)}")
        logging.info(f"Full loaded n_calibration_epochs: {get_n_epochs(raw_calibration_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(raw_calibration_trials)}")
        logging.info(f"Original time window: {window_size} epochs")
        logging.info(f"For this session, the window size will be: {window_size}")
        logging.info(f"Trained BT-LDA on the calibration data, using only the last {window_size} epochs.")
        logging.info("================================ Online ================================")
        logging.info(f"Online {log_filenames(fnon)}")
        if _have_same_preprocessing(ppcal, ppon):
            logging.info("Same preprocessing configurations as for the calibration data")
        else:
            logging.info(log_preprocessing(ppon))
        logging.info(log_feature_extraction(online_features))
        logging.info(f"n_online_trials: {len(online_trials)}")
        logging.info(f"n_online_epochs {get_n_epochs(online_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(online_trials)}")
        logging.info("Online simulation starts")
        logging.info(f"Number of online trials: {len(online_trials)}, which is {len(online_trials)/6} runs")


    ### Online simulation ------------------------------------------------------------------

    # Extract relevant data, labels and the played words
    online_trial_targets = np.array([trial[0]["Target"].events[:,2][0] % 10 for trial in online_trials]) # The target word per trial
    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]            
    online_labels = np.array(online_labels) 
    online_words = [(iteration.events[:,2]%10) for trial in online_trials for iteration in trial]
    online_words = np.array(online_words) 

    # compute distances to the decision boundary per epoch
    signed_distances_btlda = np.zeros(len(online_labels))

    # get marker info
    online_info = load_or_extract_markers(online_features.get('pickle_path'), online_trials=online_trials)
    markers2 = online_info.get('markers')

    epoch_count = 0 
    played_word_count = 0
    features = online_features.get('features')

    # word decision after a trial
    trial_predictions_btlda = np.zeros(online_trial_targets.shape)

    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            run_nr = math.trunc(t/6)+1
            logging.info(f"------------------ Run {run_nr} Trial {t%6+1}  (total trials: {t+1}/{len(online_trials)}) ------------------")
            logging.info("{epoch} | {word_id} | {BTLDA} ")

        stim_distances_btlda = np.zeros((len(trial),6))

        X_new_list = []
        y_new_list = []

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                new_x = features[epoch_count]

                # Compute signed distance of stimulus to decision boundary 
                s3 = btlda.decision_function(new_x).item()
                signed_distances_btlda[epoch_count] = s3 
                
                if log_process:
                    marker = markers2[epoch_count]
                    logging.info(f"{epoch_count} \t| {marker} \t| {s3}")

                # for word decision
                word_id = online_words[played_word_count,s] - 1 
                stim_distances_btlda[i,word_id] = s3 

                ### adaptation (sliding window)
                x = new_x
                y = online_labels[epoch_count]
                # # update X_train and y_train data
                X_new_list.append(x)
                y_new_list.append(y)

                # note that we did not update our classifier (yet)
                epoch_count+=1 

            
            played_word_count += 1

        # End of trial
        means_btlda = np.mean(stim_distances_btlda, axis=0)
        best_guess_btlda = np.argmax(means_btlda)
        trial_predictions_btlda[t] = best_guess_btlda + 1

        ### Adaptation: update our classifier after a trial has finished
        # convert to np arrays
        X_new_list_to_array = np.vstack(X_new_list)
        y_new_list_to_array = np.array(y_new_list)
        # add new trial
        X_train = np.append(X_train, X_new_list_to_array, axis=0)
        y_train = np.append(y_train, y_new_list_to_array)
        # remove oldest trial
        X_train = X_train[len(X_new_list):]
        y_train = y_train[len(y_new_list):]

        # BT-LDA
        btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
        btlda.fit(X_train,y_train)

        if log_process:
            logging.info("------------------ End of trial ------------------")
            logging.info("{real_word} | {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {}".format(online_trial_targets[t],best_guess_btlda+1))
            logging.info("Updated the BT-LDA classifier, this trial is now included in the training set & the oldest trial is removed")

    fpr_btlda, tpr_btlda, thresholds = metrics.roc_curve(online_labels,signed_distances_btlda) 

    if log_process:
        logging.info("End of online simulation")
        logging.info("------------------ Epoch-wise performance ------------------")
        logging.info(f"AUC-ROC BT-LDA: {metrics.auc(fpr_btlda, tpr_btlda):0.5f}")

    # print("------------------ Word prediction performance (per trial) ------------------")
    # print(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):0.5f}")

    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        logging.info(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):.5f} ({np.sum(trial_predictions_btlda == online_trial_targets)} correct out of {len(online_trial_targets)})")

        close_logging()
        
    performances = {
        "epoch-wise": {
            "btlda": metrics.auc(fpr_btlda, tpr_btlda),
        },
        "trial-wise": {
            "btlda": np.mean(trial_predictions_btlda == online_trial_targets),
        },
        "trial_predictions":{
            "btlda": trial_predictions_btlda,
            "true": online_trial_targets
        }
    }
    
    return performances

# This version was not used in the thesis, but computed as an experiment. Here, the window size is determined by the number of data points (epochs) of the previous session. Therefore, this window size is not fixed, but rather changes for every session
def online_window_simulation_v4(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, title_text = ""):
    """
    Online simulation withs sliding window adaptation. For every epoch, add that epoch to the training set and remove the oldest epoch from the training set. Update the classifiers only after a trial has finished.

    See online_simulation() for documentation.
    """

    # Extract information from data
    raw_calibration_trials = raw_calibration_data.get('trials')
    print("All calibration trials: ",len(raw_calibration_trials))
    print("That is {} epochs\n".format(get_n_epochs(raw_calibration_trials)))
    ppcal = raw_calibration_data.get('preprocessing')
    fncal = raw_calibration_data.get('filenames')

    online_trials = online_data.get("trials")
    ppon = online_data.get('preprocessing')
    fnon = online_data.get('filenames')

    if log_process is not None:
        start_logging(log_process)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"New log file - {timestamp}")
        logging.info("================================ Calibration ================================")
        logging.info(f"Calibration {log_filenames(fncal)}")
        logging.info(log_preprocessing(ppcal))

    # Feature extraction
    ival_bounds = calibration_features.get('fe_info').get('time_ivals')

    X_train, y_train = epoch_vectorizer_channelprime(raw_calibration_trials=raw_calibration_trials, ival_bounds=ival_bounds)

    ### Calibration -----------------------------------------------------------------------

    ## BT-LDA
    nch = (raw_calibration_trials[0][0]).info["nchan"]
    btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
    btlda.fit(X_train,y_train)


    if log_process:
        logging.info(log_feature_extraction(calibration_features, X_train.shape, y_train.shape))
        logging.info(f"n_calibration_trials: {len(raw_calibration_trials)}")
        logging.info(f"n_calibration_epochs: {get_n_epochs(raw_calibration_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(raw_calibration_trials)}")
        logging.info("Trained BT-LDA on the calibration data.")
        logging.info("================================ Online ================================")
        logging.info(f"Online {log_filenames(fnon)}")
        if _have_same_preprocessing(ppcal, ppon):
            logging.info("Same preprocessing configurations as for the calibration data")
        else:
            logging.info(log_preprocessing(ppon))
        logging.info(log_feature_extraction(online_features))
        logging.info(f"n_online_trials: {len(online_trials)}")
        logging.info(f"n_online_epochs {get_n_epochs(online_trials)}")
        logging.info(f"with the per-run iteration structure:\n{get_iteration_structure(online_trials)}")
        logging.info("Online simulation starts")
        logging.info(f"Number of online trials: {len(online_trials)}, which is {len(online_trials)/6} runs")


    ### Online simulation ------------------------------------------------------------------

    # Extract relevant data, labels and the played words
    online_trial_targets = np.array([trial[0]["Target"].events[:,2][0] % 10 for trial in online_trials]) # The target word per trial
    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]            
    online_labels = np.array(online_labels) 
    online_words = [(iteration.events[:,2]%10) for trial in online_trials for iteration in trial]
    online_words = np.array(online_words) 

    # compute distances to the decision boundary per epoch
    signed_distances_btlda = np.zeros(len(online_labels))

    # get marker info
    online_info = load_or_extract_markers(online_features.get('pickle_path'), online_trials=online_trials)
    markers2 = online_info.get('markers')

    epoch_count = 0 
    played_word_count = 0
    features = online_features.get('features')

    # word decision after a trial
    trial_predictions_btlda = np.zeros(online_trial_targets.shape)

    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            run_nr = math.trunc(t/6)+1
            logging.info(f"------------------ Run {run_nr} Trial {t%6+1}  (total trials: {t+1}/{len(online_trials)}) ------------------")
            logging.info("{epoch} | {word_id} | {BTLDA} ")

        stim_distances_btlda = np.zeros((len(trial),6))

        X_new_list = []
        y_new_list = []

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                new_x = features[epoch_count]

                # Compute signed distance of stimulus to decision boundary 
                s3 = btlda.decision_function(new_x).item()
                signed_distances_btlda[epoch_count] = s3 
                
                if log_process:
                    marker = markers2[epoch_count]
                    logging.info(f"{epoch_count} \t| {marker} \t| {s3}")

                # for word decision
                word_id = online_words[played_word_count,s] - 1 
                stim_distances_btlda[i,word_id] = s3 

                ### adaptation (sliding window)
                x = new_x
                y = online_labels[epoch_count]
                # # update X_train and y_train data
                X_new_list.append(x)
                y_new_list.append(y)
                # X_train = np.append(X_train,x, axis=0)
                # y_train = np.append(y_train,y)
                # # remove oldest data point
                # X_train = X_train[1:]
                # y_train = y_train[1:]

                # note that we did not update our classifier (yet)
                epoch_count+=1 

            
            played_word_count += 1

        # End of trial
        means_btlda = np.mean(stim_distances_btlda, axis=0)
        best_guess_btlda = np.argmax(means_btlda)
        trial_predictions_btlda[t] = best_guess_btlda + 1

        ### Adaptation: update our classifier after a trial has finished
        # convert to np arrays
        X_new_list_to_array = np.vstack(X_new_list)
        y_new_list_to_array = np.array(y_new_list)
        # add new trial
        X_train = np.append(X_train, X_new_list_to_array, axis=0)
        y_train = np.append(y_train, y_new_list_to_array)
        # remove oldest trial
        X_train = X_train[len(X_new_list):]
        y_train = y_train[len(y_new_list):]

        # BT-LDA
        btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
        btlda.fit(X_train,y_train)

        if log_process:
            logging.info("------------------ End of trial ------------------")
            logging.info("{real_word} | {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {}".format(online_trial_targets[t],best_guess_btlda+1))
            logging.info("Updated the BT-LDA classifier, this trial is now included in the training set & the oldest trial is removed")

    fpr_btlda, tpr_btlda, thresholds = metrics.roc_curve(online_labels,signed_distances_btlda) 

    if log_process:
        logging.info("End of online simulation")
        logging.info("------------------ Epoch-wise performance ------------------")
        logging.info(f"AUC-ROC BT-LDA: {metrics.auc(fpr_btlda, tpr_btlda):0.5f}")

    # print("------------------ Word prediction performance (per trial) ------------------")
    # print(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):0.5f}")

    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        logging.info(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):.5f} ({np.sum(trial_predictions_btlda == online_trial_targets)} correct out of {len(online_trial_targets)})")

        close_logging()
        
    performances = {
        "epoch-wise": {
            "btlda": metrics.auc(fpr_btlda, tpr_btlda),
        },
        "trial-wise": {
            "btlda": np.mean(trial_predictions_btlda == online_trial_targets),
        },
        "trial_predictions":{
            "btlda": trial_predictions_btlda,
            "true": online_trial_targets
        }
    }
    
    return performances