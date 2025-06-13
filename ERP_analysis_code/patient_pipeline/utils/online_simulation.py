import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from toeplitzlda.classification import ToeplitzLDA
from utils.feature_extraction import get_jumping_means, epoch_vectorizer_channelprime, load_or_extract_markers
from utils.preprocessing import _have_same_preprocessing, get_n_epochs, get_iteration_structure
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import logging
import matplotlib as mpl

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

def online_simulation(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, title_text = ""):
    """
    Online simulation of transfer fixed sLDA, and BT-LDA (with no within-session updating)

    All classifiers are trained on the calibration trials. Then, for the online trials in the online simulation, they will predict the label of every played word (epoch) within a trial and decode the target word at the end of every online trial. At the end of the simulation the predictions will be compared against the real labels and real target words. An epoch-wise (label prediction) and trial-wise (target word prediction) will be logged and plotted.

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

    slda = make_pipeline(LDA(solver='lsqr', shrinkage='auto'),) # SLDA
    slda.fit(X,y)
    nch = (raw_calibration_trials[0][0]).info["nchan"]
    btlda = make_pipeline(ToeplitzLDA(n_channels=nch),) # BTLDA
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

    # markers2 = np.zeros(len(online_labels), np.int8)
    # for trial in online_trials:
    #     for iteration in trial:
    #         for e, epoch in enumerate(iteration):
    #             markers2[epoch_c] = iteration[e].events[0,2]
    #             #print(iteration[e].events[0,2])
    #             epoch_c+=1

    online_info = load_or_extract_markers(online_features.get('pickle_path'), online_trials=online_trials)
    markers2 = online_info.get('markers')

    # online_trial_targets: the target word per trial. e.g. [4 5 6 2 3 1 1 6 2 4]
    # length = n_trials, e.g. 150

    # online_labels: labels of all epochs. e.g. [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    # length = all epochs (= n_trials * n_iterations per trial * n_epochs per iteration), e.g. 8244
   
    # online_words: The word ID sequence that is presented per iteration. Note that the order differs between iterations.
    # print(online_words[:6]) # e.g. [[1 6 2 3 5 4]
                              #       [6 3 1 2 4 5]
                              #       [6 3 5 2 1 4]
                              #       [3 1 2 6 4 5]
                              #       [1 6 5 2 4 3]
                              #       [5 4 1 6 2 3]]
    # shape = (n_trials * avg_n_iterations_per_trial, 6), e.g. (1374, 6)

    # store distances to the decision boundary per epoch
    signed_distances_slda = np.zeros(len(online_labels))
    signed_distances_btlda = np.zeros(len(online_labels))

    epoch_count = 0 
    played_word_count = 0
    features = online_features.get('features')

    # word decision after a trial
    trial_predictions_slda = np.zeros(online_trial_targets.shape)
    trial_predictions_btlda = np.zeros(online_trial_targets.shape)


    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            run_nr = math.trunc(t/6)+1
            logging.info(f"------------------ Run {run_nr} Trial {t%6+1}  (total trials: {t+1}/{len(online_trials)}) ------------------")
            logging.info("{epoch} | {word_id} | {SLDA} \t\t\t\t| {BTLDA} ")

        stim_distances_slda = np.zeros((len(trial),6))
        stim_distances_btlda = np.zeros((len(trial),6))

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                x = features[epoch_count]

                s2 = (slda.decision_function(x))[0]
                signed_distances_slda[epoch_count] = s2 # Compute signed distance of stimulus to decision boundary

                s3 = btlda.decision_function(x).item()
                signed_distances_btlda[epoch_count] = s3 # Compute signed distance of stimulus to decision boundary
                
                if log_process:
                    #marker = iteration[s].events[:,2]
                    marker = markers2[epoch_count]
                    logging.info(f"{epoch_count} \t| {marker} \t| {s2} \t| {s3}")

                # for word decision
                word_id = online_words[played_word_count,s] - 1 # get the word id of the current epoch/stimulus s
                stim_distances_slda[i,word_id] = s2
                stim_distances_btlda[i,word_id] = s3 
                epoch_count+=1
            
            played_word_count += 1

        # End of trial
        means_slda = np.mean(stim_distances_slda, axis=0) # get the mean distance for each word in the trial
        means_btlda = np.mean(stim_distances_btlda, axis=0) # get the mean distance for each word in the trial

        best_guess_slda = np.argmax(means_slda) # predict the word
        best_guess_btlda = np.argmax(means_btlda) # predict the word

        trial_predictions_slda[t] = best_guess_slda + 1 # convert to correct word id
        trial_predictions_btlda[t] = best_guess_btlda + 1 # convert to correct word id
 
        # For p-values
        # best_distances_slda = stim_distances_slda[:, best_guess_slda].flatten()
        # best_distances_btlda = stim_distances_btlda[:, best_guess_btlda].flatten()
        #
        # not_best_distances = stim_distances[:,np.arange(stim_distances.shape[1])!=best_guess].flatten()
        # t_score, p = stats.ttest_ind(best_distances, not_best_distances, equal_var = False)
        #
        # print("Trial %d target prediction: word %d with p-value of %0.6f" % (t, best_guess+1, p)) 

        if log_process:
            logging.info("------------------ End of trial ------------------")
            logging.info("{real_word} | {SLDA_prediction} \t| {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {} \t\t\t\t\t| {} ".format(online_trial_targets[t],best_guess_slda+1,best_guess_btlda+1))

    print("------------------ Epoch-wise performance ------------------")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18,6)) # 1 row, 2 cols

    fpr_slda, tpr_slda, thresholds = metrics.roc_curve(online_labels,signed_distances_slda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_slda, tpr = tpr_slda)
    auc_fig.plot(ax=axes[0],color='orange',label="AUC")
    axes[0].plot([0, 1],[0,1], '--', color="gray", label="area = 0.5")
    axes[0].legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_slda, tpr_slda), 'area = 0.5'], loc="lower right")
    axes[0].set_title("AUC-ROC of Transfer Fixed sLDA [online]")

    fpr_btlda, tpr_btlda, thresholds = metrics.roc_curve(online_labels,signed_distances_btlda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_btlda, tpr = tpr_btlda)
    auc_fig.plot(ax=axes[1],color='orange',label="AUC")
    axes[1].plot([0, 1],[0,1], '--', color='gray', label="area = 0.5")
    axes[1].legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_btlda, tpr_btlda), 'area = 0.5'], loc="lower right")
    axes[1].set_title("AUC-ROC of Transfer Fixed BT-LDA [online]")
    
    plt.suptitle(f"Online epoch-wise performance of all transfer fixed classifiers (no within-session updating) - "+title_text)
    plt.show()

    if log_process:
        logging.info("End of online simulation")
        logging.info("------------------ Epoch-wise performance ------------------")
        logging.info(f"AUC-ROC SLDA: {metrics.auc(fpr_slda, tpr_slda):.5f}")
        logging.info(f"AUC-ROC BT-LDA: {metrics.auc(fpr_btlda, tpr_btlda):0.5f}")
    
    print("------------------ Word prediction performance (per trial) ------------------")
    print(f"Accuracy SLDA: {np.mean(trial_predictions_slda == online_trial_targets):.5f}")
    print(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):0.5f}")

    # For p-values
    # plot_distribution_comparison(not_best_distances, best_distances) (see assignment 07)

    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        logging.info(f"Accuracy SLDA: {np.mean(trial_predictions_slda == online_trial_targets):.5f} ({np.sum(trial_predictions_slda == online_trial_targets)} correct out of {len(online_trial_targets)})")
        logging.info(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):.5f} ({np.sum(trial_predictions_btlda == online_trial_targets)} correct out of {len(online_trial_targets)})")

        close_logging()
        
        # Log the final coefficient w and intercept b
        #
        #w_lda = ((ldaclf.get_params().get("lineardiscriminantanalysis")).coef_) # obtain w vector
        #w_slda = (slda.get_params().get("lineardiscriminantanalysis").coef_) # obtain w vector
        #btlda_param1 = ((btlda.get_params().get("toeplitzlda")).coef_) # cov matrix?
        #btlda_param2 = (btlda.get_params().get("toeplitzlda").intercept_) # what is this?
        #logging.info("------- LDA -------")
        #logging.info(f"w: {w_lda} \t| b: {0}")
        #logging.info("------- SLDA -------")
        #logging.info(f"w: {w_slda} \t| b: {0}")
        #logging.info("------- BTLDA -------")
        #logging.info(f"w and b is still to be obtained (I have to solve this)")
        #logging.info(f"__coef__: {btlda_param1} \t| __intercept__: {btlda_param2}")

    performances = {
        "epoch-wise": {
            "slda": metrics.auc(fpr_slda, tpr_slda),
            "btlda": metrics.auc(fpr_btlda, tpr_btlda),
        },
        "trial-wise": {
            "slda": np.mean(trial_predictions_slda == online_trial_targets),
            "btlda": np.mean(trial_predictions_btlda == online_trial_targets),
        },
        "trial_predictions":{
            "slda": trial_predictions_slda,
            "btlda": trial_predictions_btlda,
            "true": online_trial_targets
        }
    }
    
    return performances

def online_window_simulation_v3(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, title_text = ""):
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

def online_window_simulation_v1(raw_calibration_data:dict, online_data:dict, calibration_features:dict, online_features:dict, log_process=None, title_text = ""):
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

    ## Shrinkage LDA
    slda = make_pipeline(LDA(solver='lsqr', shrinkage='auto'),)
    slda.fit(X_train,y_train)
    ## BT-LDA
    nch = (raw_calibration_trials[0][0]).info["nchan"]
    btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
    btlda.fit(X_train,y_train)


    if log_process:
        logging.info(log_feature_extraction(calibration_features, X_train.shape, y_train.shape))
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
    online_trial_targets = np.array([trial[0]["Target"].events[:,2][0] % 10 for trial in online_trials]) # The target word per trial
    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]            
    online_labels = np.array(online_labels) 
    online_words = [(iteration.events[:,2]%10) for trial in online_trials for iteration in trial]
    online_words = np.array(online_words) 

    # compute distances to the decision boundary per epoch
    signed_distances_slda = np.zeros(len(online_labels))
    signed_distances_btlda = np.zeros(len(online_labels))

    # get marker info
    online_info = load_or_extract_markers(online_features.get('pickle_path'), online_trials=online_trials)
    markers2 = online_info.get('markers')

    epoch_count = 0 
    played_word_count = 0
    features = online_features.get('features')

    # word decision after a trial
    trial_predictions_slda = np.zeros(online_trial_targets.shape)
    trial_predictions_btlda = np.zeros(online_trial_targets.shape)

    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            run_nr = math.trunc(t/6)+1
            logging.info(f"------------------ Run {run_nr} Trial {t%6+1}  (total trials: {t+1}/{len(online_trials)}) ------------------")
            logging.info("{epoch} | {word_id} | {SLDA} \t\t\t\t| {BTLDA} ")

        stim_distances_slda = np.zeros((len(trial),6))
        stim_distances_btlda = np.zeros((len(trial),6))

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):

                # Obtain x (of a single epoch)
                new_x = features[epoch_count]

                # Compute signed distance of stimulus to decision boundary 
                s2 = (slda.decision_function(new_x))[0]
                signed_distances_slda[epoch_count] = s2
                s3 = btlda.decision_function(new_x).item()
                signed_distances_btlda[epoch_count] = s3 
                
                if log_process:
                    marker = markers2[epoch_count]
                    logging.info(f"{epoch_count} \t| {marker} \t| {s2} \t| {s3}")

                # for word decision
                word_id = online_words[played_word_count,s] - 1 
                stim_distances_slda[i,word_id] = s2
                stim_distances_btlda[i,word_id] = s3 

                ### adaptation (sliding window)
                x = new_x
                y = online_labels[epoch_count]
                # update X_train and y_train data
                X_train = np.append(X_train,x, axis=0)
                y_train = np.append(y_train,y)
                # if growing is False:
                #     X_train = X_train[1:]
                #     y_train = y_train[1:]

                # note that we did not update our classifier (yet)
                epoch_count+=1 

            
            played_word_count += 1

        # End of trial
        means_slda = np.mean(stim_distances_slda, axis=0) 
        means_btlda = np.mean(stim_distances_btlda, axis=0)

        best_guess_slda = np.argmax(means_slda)
        best_guess_btlda = np.argmax(means_btlda)

        trial_predictions_slda[t] = best_guess_slda + 1
        trial_predictions_btlda[t] = best_guess_btlda + 1

        ### Adaptation: update our classifier after a trial has finished
        # sLDA
        slda = make_pipeline(LDA(solver='lsqr', shrinkage='auto'),)
        slda.fit(X_train,y_train)
        # BT-LDA
        btlda = make_pipeline(ToeplitzLDA(n_channels=nch),)
        btlda.fit(X_train,y_train)

        if log_process:
            logging.info("------------------ End of trial ------------------")
            logging.info("{real_word} | {SLDA_prediction} \t| {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {} \t\t\t\t\t| {} ".format(online_trial_targets[t],best_guess_slda+1,best_guess_btlda+1))
            logging.info("Updated all classifiers, this trial is now included in the training set")

    print("------------------ Epoch-wise performance ------------------")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18,6)) # 1 row, 2 columns 

    fpr_slda, tpr_slda, thresholds = metrics.roc_curve(online_labels,signed_distances_slda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_slda, tpr = tpr_slda)
    auc_fig.plot(ax=axes[0],color='orange',label="AUC")
    axes[0].plot([0, 1],[0,1], '--', color="gray", label="area = 0.5")
    axes[0].legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_slda, tpr_slda), 'area = 0.5'], loc="lower right")
    axes[0].set_title("AUC-ROC of Window sLDA [online] [sw]")

    fpr_btlda, tpr_btlda, thresholds = metrics.roc_curve(online_labels,signed_distances_btlda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_btlda, tpr = tpr_btlda)
    auc_fig.plot(ax=axes[1],color='orange',label="AUC")
    axes[1].plot([0, 1],[0,1], '--', color='gray', label="area = 0.5")
    axes[1].legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_btlda, tpr_btlda), 'area = 0.5'], loc="lower right")
    axes[1].set_title("AUC-ROC of Window BT-LDA [online] [sw]")
    
    plt.suptitle(f"Online epoch-wise performance using sliding window updating - "+title_text)
                 
    # if growing:
    #     plt.suptitle(f"Online epoch-wise performance of all adaptive classifiers using growing window updating - "+title_text)
    # else:
    #     plt.suptitle(f"Online epoch-wise performance of all adaptive classifiers using sliding window updating - "+title_text)

    plt.show()

    if log_process:
        logging.info("End of online simulation")
        logging.info("------------------ Epoch-wise performance ------------------")
        logging.info(f"AUC-ROC SLDA: {metrics.auc(fpr_slda, tpr_slda):.5f}")
        logging.info(f"AUC-ROC BT-LDA: {metrics.auc(fpr_btlda, tpr_btlda):0.5f}")

    print("------------------ Word prediction performance (per trial) ------------------")
    print(f"Accuracy SLDA: {np.mean(trial_predictions_slda == online_trial_targets):.5f}")
    print(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):0.5f}")

    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        logging.info(f"Accuracy SLDA: {np.mean(trial_predictions_slda == online_trial_targets):.5f} ({np.sum(trial_predictions_slda == online_trial_targets)} correct out of {len(online_trial_targets)})")
        logging.info(f"Accuracy BT-LDA: {np.mean(trial_predictions_btlda == online_trial_targets):.5f} ({np.sum(trial_predictions_btlda == online_trial_targets)} correct out of {len(online_trial_targets)})")

        close_logging()
        
    performances = {
        "epoch-wise": {
            "slda": metrics.auc(fpr_slda, tpr_slda),
            "btlda": metrics.auc(fpr_btlda, tpr_btlda),
        },
        "trial-wise": {
            "slda": np.mean(trial_predictions_slda == online_trial_targets),
            "btlda": np.mean(trial_predictions_btlda == online_trial_targets),
        },
        "trial_predictions":{
            "slda": trial_predictions_slda,
            "btlda": trial_predictions_btlda,
            "true": online_trial_targets
        }
    }
    
    return performances