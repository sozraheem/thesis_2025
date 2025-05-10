import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from toeplitzlda.classification import ToeplitzLDA
from utils.feature_extraction import get_jumping_means, epoch_vectorizer_channelprime

def online_simulation(raw_calibration_trials, online_trials, ival_bounds = np.array([0.1, 0.2, 0.3, 0.4, 0.5]), log_process=None):

    if log_process is not None:
        
        # this was needed in order to create a log file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=log_process,
            encoding="utf-8",
            filemode="w", # 'a' to not overwrite current log, 'w' to overwrite. This setting can be changed later
            level=logging.DEBUG)

        logging.info("New log file")

    # Feature extraction
    clf_ival_boundaries = ival_bounds
    X, y = epoch_vectorizer_channelprime(raw_calibration_trials=raw_calibration_trials, ival_bounds=ival_bounds)

    ### Calibration -----------------------------------------------------------------------

    ### LDA
    ldaclf = make_pipeline(LDA(),)
    ldaclf.fit(X,y)

    ### Shrinkage LDA
    slda = make_pipeline(LDA(solver='lsqr', shrinkage='auto'),)
    slda.fit(X,y)

    ### BT-LDA
    nch = (raw_calibration_trials[0][0]).info["nchan"]
    btlda = make_pipeline(
        ToeplitzLDA(n_channels=nch),
    )
    btlda.fit(X,y)

    if log_process:
        logging.info("Trained all three classifiers on the calibration data.")
        logging.info("Online simulation starts")

    ### Online simulation ------------------------------------------------------------------

    # Extract relevant data, labels and the played words

    # Using list comprehension
    online_trial_targets = np.array([trial[0]["Target"].events[:,2][0] % 10 for trial in online_trials]) # The target word per trial
    online_labels = [(1 if event > 107 else 0) for trial in online_trials for iteration in trial for event in iteration.events[:,2]]            
    online_labels = np.array(online_labels) # conversion to np array is maybe not even needed
    online_words = [(iteration.events[:,2]%10) for trial in online_trials for iteration in trial]
    online_words = np.array(online_words) # conversion to np array is maybe not even needed

    # online_trial_targets: the target word per trial. e.g. [4 5 6 2 3 1 1 6 2 4]
    # length = n_trials, e.g. 150

    # online_labels: labels of all epochs. e.g. [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    # length = all epochs (= n_trials * n_iterations per trial * n_epochs per iteration) # e.g. 8244
   
    # online_words: The word ID sequence that is presented per iteration. Note that the order differs between iterations.
    # print(online_words[:6]) # e.g. [[1 6 2 3 5 4]
                              #       [6 3 1 2 4 5]
                              #       [6 3 5 2 1 4]
                              #       [3 1 2 6 4 5]
                              #       [1 6 5 2 4 3]
                              #       [5 4 1 6 2 3]]
    print(online_words.shape) # (n_trials * avg_n_iterations_p_trial, 6) e.g. (1374, 6)

    if log_process:
        logging.info("Number of online trials: {}, which is {} runs".format(len(online_trials), len(online_trials)/6))

    # compute distances to the decision boundary per epoch
    signed_distances_lda = np.zeros(len(online_labels))
    signed_distances_slda = np.zeros(len(online_labels))
    signed_distances_btlda = np.zeros(len(online_labels))

    count = 0 
    played_word_count = 0

    # word decision after a trial
    trial_predictions_lda = np.zeros(online_trial_targets.shape)
    trial_predictions_slda = np.zeros(online_trial_targets.shape)
    trial_predictions_btlda = np.zeros(online_trial_targets.shape)


    for t, trial in enumerate(online_trials):
        print("trial {}/{}".format(t, len(online_trials)))
        if log_process:
            logging.info("------------------ Run {} Trial {}  (total trials: {}/{}) ------------------".format(math.trunc(t/6)+1,(t+1)%6+1, t+1, len(online_trials)))
            logging.info("{epoch} \t| {word_id} \t| {LDA} \t| {SLDA} \t| {BTLDA} ")

        stim_distances_lda = np.zeros((len(trial),6))
        stim_distances_slda = np.zeros((len(trial),6))
        stim_distances_btlda = np.zeros((len(trial),6))

        for i, iteration in enumerate(trial):
            for s, stimulus in enumerate(iteration):
                s1 = (ldaclf.decision_function(get_jumping_means(iteration[s],clf_ival_boundaries).transpose(0,2,1).flatten().reshape(1,-1)))[0]
                signed_distances_lda[count] = s1 # Compute signed distance of stimulus to decision boundary

                s2 = (slda.decision_function(get_jumping_means(iteration[s],clf_ival_boundaries).transpose(0,2,1).flatten().reshape(1,-1)))[0]
                signed_distances_slda[count] = s2 # Compute signed distance of stimulus to decision boundary

                s3 = btlda.decision_function(get_jumping_means(iteration[s],clf_ival_boundaries).transpose(0,2,1).flatten().reshape(1,-1)).item()
                signed_distances_btlda[count] = s3 # Compute signed distance of stimulus to decision boundary
                
                if log_process:
                    logging.info("{} \t| {} \t| {} \t| {} \t| {}".format(count, iteration[s].events[:,2], s1, s2, s3))

                # for word decision
                played_word = online_words[played_word_count,s] - 1 # convert to index
                stim_distances_lda[i,played_word] = s1 # order computed distances according to word id
                stim_distances_slda[i,played_word] = s2
                stim_distances_btlda[i,played_word] = s3 
                count+=1

                # Important note during debugging
                # btlda.decision_function returns an nd array of shape (). To access its value, you have to call .item() additionally, instead of taking the first element via [0] (as done for lda and slda)
            
            played_word_count += 1

        means_lda = np.mean(stim_distances_lda, axis=0) # get the mean distance for each word in the trial
        means_slda = np.mean(stim_distances_slda, axis=0) # get the mean distance for each word in the trial
        means_btlda = np.mean(stim_distances_btlda, axis=0) # get the mean distance for each word in the trial

        best_guess_lda = np.argmax(means_lda) # predict the word
        best_guess_slda = np.argmax(means_slda) # predict the word
        best_guess_btlda = np.argmax(means_btlda) # predict the word

        # For p-values
        # best_distances_lda = stim_distances_lda[:, best_guess_lda].flatten()
        # best_distances_slda = stim_distances_slda[:, best_guess_slda].flatten()
        # best_distances_btlda = stim_distances_btlda[:, best_guess_btlda].flatten()

        #not_best_distances = stim_distances[:,np.arange(stim_distances.shape[1])!=best_guess].flatten()
        #t_score, p = stats.ttest_ind(best_distances, not_best_distances, equal_var = False)

        trial_predictions_lda[t] = best_guess_lda + 1
        trial_predictions_slda[t] = best_guess_slda + 1
        trial_predictions_btlda[t] = best_guess_btlda + 1

        #print("Trial %d target prediction: word %d with p-value of %0.6f" % (t, best_guess+1, p)) 
        if log_process:
            logging.info("------------------ End of trial ------------------".format(math.trunc(t/6)+1,t+1))
            logging.info("{real_word} \t| {LDA_prediction} \t| {SLDA_prediction} \t| {BTLDA_prediction} ")
            logging.info("{} \t\t\t| {} \t\t\t\t| {} \t\t\t\t\t| {} ".format(online_trial_targets[t],best_guess_lda+1,best_guess_slda+1,best_guess_btlda+1))

    print("------------------ Epoch-wise performance ------------------".format(math.trunc(t/6)+1,t+1))

    fpr_lda, tpr_lda, thresholds = metrics.roc_curve(online_labels,signed_distances_lda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_lda, tpr = tpr_lda)
    auc_fig.plot(label="AUC")
    plt.plot([0, 1],[0,1], '--', label="area = 0.5")
    plt.legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_lda, tpr_lda), 'area = 0.5'], loc="lower right")
    plt.title("AUC-ROC Curve of the LDA classifier [online]")
    plt.show()


    fpr_slda, tpr_slda, thresholds = metrics.roc_curve(online_labels,signed_distances_slda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_slda, tpr = tpr_slda)
    auc_fig.plot(label="AUC")
    plt.plot([0, 1],[0,1], '--', label="area = 0.5")
    plt.legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_slda, tpr_slda), 'area = 0.5'], loc="lower right")
    plt.title("AUC-ROC Curve of the sLDA classifier [online]")
    plt.show()


    fpr_btlda, tpr_btlda, thresholds = metrics.roc_curve(online_labels,signed_distances_btlda) 
    auc_fig = metrics.RocCurveDisplay(fpr=fpr_btlda, tpr = tpr_btlda)
    auc_fig.plot(label="AUC")
    plt.plot([0, 1],[0,1], '--', label="area = 0.5")
    plt.legend(['ROC curve (area = %0.5f)' % metrics.auc(fpr_btlda, tpr_btlda), 'area = 0.5'], loc="lower right")
    plt.title("AUC-ROC Curve of the BT-LDA classifier [online]")
    plt.show()

    if log_process:
        logging.info("------------------ Epoch-wise performance ------------------")
        logging.info("AUC-ROC LDA: %0.5f" % metrics.auc(fpr_lda, tpr_lda))
        logging.info("Accuracy SLDA: %0.5f" % metrics.auc(fpr_slda, tpr_slda))
        logging.info("Accuracy BT-LDA: %0.5f" % metrics.auc(fpr_btlda, tpr_btlda))

    # word prediction
    
    # plot_distribution_comparison(not_best_distances, best_distances)
    print("------------------ Word prediction performance (per trial) ------------------")
    print("Accuracy LDA: %0.5f" % np.mean(trial_predictions_lda == online_trial_targets))
    print("Accuracy SLDA: %0.5f" % np.mean(trial_predictions_slda == online_trial_targets))
    print("Accuracy BT-LDA: %0.5f" % np.mean(trial_predictions_btlda == online_trial_targets))

    if log_process:
        logging.info("------------------ Word prediction performance (per trial) ------------------")
        logging.info("Accuracy LDA: %0.5f" % np.mean(trial_predictions_lda == online_trial_targets))
        logging.info("Accuracy SLDA: %0.5f" % np.mean(trial_predictions_slda == online_trial_targets))
        logging.info("Accuracy BT-LDA: %0.5f" % np.mean(trial_predictions_btlda == online_trial_targets))

        #close_logging()
        

    return online_trial_targets

# def close_logging():
#     # close and remove all handlers
#     logger = logging.getLogger()
#     for handler in logger.handlers[:]:
#         handler.close()
#         logger.removeHandler(handler)