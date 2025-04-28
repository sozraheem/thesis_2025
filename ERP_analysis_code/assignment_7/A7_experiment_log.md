# A7 Experiments/Modifications Log

To do: 
- assign IDs to experiments?
- add instructions on how to navigate through this file
- make images smaller?

Experiment ideas:
- train_test_split with different test sizes (20%, 30%)
- cv instead of train_test_split [calibration]
- make classification per trial harder by changing feature extraction methods [online]
-
- BT-LDA with different time intervals
- Test sliding window on LDA with different window sizes
- Test sliding window on sLDA and BT-LDA
- Implement forgetting strategies from scratch (see notes) 
- See to do lists of previous experiments & notes


Overview 

- 🔧**25/04/2025_MDF_2:** Use K-folds cross-validation **[calibration]** ✏️
- 📋**25/04/2025_Exp_7:** Compare K-folds cv with train_test_split **[calibration]** ✏️
- 📋**25/04/2025_Exp_6:** Compare AUC of LDA vs sLDA vs BTLDA using different test_size values **[calibration]** ✏️
- 📙**25/04/2025_Note_4:** Current train_test_split should change **[calibration]** 
- 
- 📋**21/04/2025_Exp_5:** Implement first draft adaptive LDA: sliding window with different step sizes **[online]** ✏️
- 📋**21/04/2025_Exp_4:** Compare static LDA vs SLDA vs BT-LDA (using AUC-ROC curves, per epoch) **[online]** ✏️
- 📋**19/04/2025_Exp_3:** Compare AUC-scores of LDA vs SLDA vs BT-LDA **[calibration]** ✏️
- 📙**18/04/2025_Note_3:** How accuracy is measured **[calibration]** (2/2) 
- 📋**14/04/2025_Exp_2:** Use TimeSeriesSplit **[calibration]** 
- 📙**14/04/2025_Note_2:** sklearn's TimeSeriesSplit: parameter max_train_size 
- 📙**13/04/2025_Note_1:** How accuracy is measured **[calibration]** (1/2)  
- 🔧**11/04/2025_MDF_1:** Turn off baseline correction 
- 📋**11/04/2025_Exp_1:** Effect of baseline correction on LDA **[calibration]** 

Legend
-
- 📋**day/month/year_Exp:** Experiment: *try out different things and look at the results, but do not change the the code*
- 📙**day/month/year_Note:** Notes
- 🔧**day/month/year_MDF:** Modifications: *change the code. This new setting holds for all experiments conducted after this modification*
- **[calibration]:** Calibration part. Here we use 12 calibration_trials (1080 epochs) and we split this into a train set and test set
- **[online]:** Online simulation part. Here we use 24 online_trials (2160 epochs) and the already trained classifier (using the train set of the calibration part)
- ✏️ : in progress


## 📅 New date template

### 📋 Exp / 📙 Note / 🔧 MDF 1: [Title]

**Goal**: ...

**Change:** ...

**Results:** ...

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---
## 📅 25/04/2025

### 🔧 MDF 1: Use K-folds cross-validation [calibration]

**Goal**: ...

**Change:** ...

**Results:** ...

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---

### 📋 Exp 7: Compare K-folds cv with train_test_split [calibration]

**Goal**: ...

**Change:** ...

**Results:** ...

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---

### 📋 Exp 6: Compare AUC of LDA vs sLDA vs BTLDA using different test_size values [calibration]

**Goal**: Use different test sizes when splitting the calibration data with `train_test_split`. Then check the AUC scores of LDA, sLDA and BT-LDA

**Change:** `train_test_split` with `test_size = 10%` --> `test_size = 20%` and `test_size = 30%`

**Results:** 

Using `test_size = 0.1`
```
LDA scores with channel prime data
roc_auc:  0.8197530864197531
bal_acc_auc:  0.7333333333333334

sLDA scores with channel prime data
roc_auc:  0.8117283950617283
bal_acc_auc:  0.6444444444444445

BT LDA scores with channel prime data
roc_auc:  0.8253086419753086
bal_acc_auc:  0.65
```

Using `test_size = 0.2`
```
LDA scores with channel prime data
roc_auc:  0.817746913580247
bal_acc_auc:  0.7166666666666667

sLDA scores with channel prime data
roc_auc:  0.8265432098765431
bal_acc_auc:  0.575

BT LDA scores with channel prime data
roc_auc:  0.8294753086419753
bal_acc_auc:  0.6027777777777777
```
Using `test-size = 0.3`
```
LDA scores with channel prime data
roc_auc:  0.7780521262002743
bal_acc_auc:  0.6537037037037037

sLDA scores with channel prime data
roc_auc:  0.8282578875171468
bal_acc_auc:  0.5685185185185185

BT-LDA scores with channel prime data
roc_auc:  0.8244170096021949
bal_acc_auc:  0.5685185185185185
```
![test_size_0.3](exp_6_test_size_0.3.png)
**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---


### 📙 Note 4: Current train_test_split should change **[calibration]**

**Notes:** 
- The current method used to split the calibration data into a train and test set to then measure the AUC score, is with `train_test_split` 
- This method was discussed earlier in Note 1, where it became clear that this method is not really robust. Currently, we are splitting the dataset of 1080 epochs (i.e., the 12 `calibration_trials`) only once, with a train/test ratio of 90/10. Especially the size of the test set (+/- 100 epochs) is so small that the AUC score will not be reliable.

**To do:** 
- In Exp 6 we will use a train/test ratio of 80/20 and 70/30 and measure the AUC score.
- In Exp 7 we will replace the single `train_test_split` by a K-fold cross validation

---

## 📅21/04/2025

### 📋 Exp 5: Implement first draft adaptive LDA: sliding window with different step sizes **[online]**

**Goal**: Compare sliding window adaptation with different step sizes: update lda every 100, 10 and 1 epoch(s).

**Results:** 

![AUC_online_static_lda](images/ex4_auc_static_lda.png)
*Figure 1. AUC-ROC static LDA online*

![AUC_online_adaptive_lda_sw_100](images/ex4_auc_adaptive_lda_sw_100.png)
*Figure 2. AUC-ROC adaptive LDA online - sliding window with step size 100*

![AUC_online_adaptive_lda_sw_10](images/ex4_auc_adaptive_lda_sw_10.png)
*Figure 3. AUC-ROC adaptive LDA online - sliding window with step size 10*

![AUC_online_adaptive_lda_sw_1](images/ex4_auc_adaptive_lda_sw_1.png)
*Figure 3. AUC-ROC adaptive LDA online - sliding window with step size 1*

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---

### 📋 Exp 4: Compare static LDA vs SLDA vs BT-LDA (using AUC-ROC curves, per epoch) **[online]** 

**Goal**: ...

**Change:** ...

**Results:** 

![AUC_online_static_lda](images/ex4_auc_static_lda.png)
*Figure 1. AUC-ROC static LDA online*

![AUC_online_static_slda](images/ex4_auc_static_slda.png)
*Figure 2. AUC-ROC static sLDA online*

![AUC_online_static_btlda](images/ex4_auc_static_btlda.png)
*Figure 3. AUC-ROC static BT-LDA online*

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---

## 📅 19/04/2025

### 📋 Exp 3b: Compare AUC scores of LDA vs SLDA vs BT-LDA using another method **[calibration]**

**Goal**: Compare ROC AUC scores of LDA, sLDA and BT-LDA on the calibration data, using the evaluation method of assignment 7, ex. 3

**Results:** ... (Add plots)

**Preprocessing/Settings:** Same as 19/04/2025_Exp_1a

**Notes:** ...

**To do:** ...

### 📋 Exp 3a: Compare AUC scores of LDA vs SLDA vs BT-LDA using Jan's method **[calibration]**

**Goal**: Compare ROC AUC scores of LDA, sLDA and BT-LDA on the calibration data. The evaluation method of Jan's `example_toeplitz_lda_simply.py` script was used.

**Change:** Some things had to be changed as required for BT-LDA. See 'Preprocessing/Settings'. 

**Results:** 

```
LDA scores with channel prime data
roc_auc:  0.8197530864197531
bal_acc_auc:  0.7333333333333334

sLDA scores with channel prime data
roc_auc:  0.8117283950617283
bal_acc_auc:  0.6444444444444445

BT LDA scores with channel prime data
roc_auc:  0.8253086419753086
bal_acc_auc:  0.65
```

**Preprocessing/Settings:** 
- The data had to be reshaped to be channel-prime (required for BT-LDA). This did not change the ROC curve / AUC score of LDA or sLDA. See 19/04/2025_Exp_2 for the results when turning off the channel-prime order

- Preprocessing:
    - Bandpass-filtering = (0.5, 16 Hz)
    - `raw.filter(*filter_band, method="iir")`
    - Baseline interval = `None` 
    - Sampling rate 1000 Hz --> down sampled to 100 Hz
    - Outlier rejection: None 

- Epochs:
    - tmin = -0.2 s 
    - tmax = 1.0 s 
    - 63 EEG channels x 4 time intervals = 252 features
    - 1080 epochs used (out of 3240 epochs in total; see notes on dataset)

- Evaluation (how AUC was measured):
```
# Evaluation of Jan's simple toeplitz example script

### LDA

clf_lda = make_pipeline(
    LDA(),
)
clf_lda.fit(X_train,y_train)

y_df = clf_lda.decision_function(X_test)
roc_auc_lda = roc_auc_score(y_test, y_df)
y_pred = clf_lda.predict(X_test)
bal_acc_auc_lda = balanced_accuracy_score(y_test, y_pred)

print("LDA scores with channel prime data")
print("roc_auc: ",roc_auc_lda)
print("bal_acc_auc: ",bal_acc_auc_lda)

### sLDA

clf_slda = make_pipeline(
    LDA(solver='lsqr',
        shrinkage='auto'),
)
clf_slda.fit(X_train,y_train)

y_df = clf_slda.decision_function(X_test)
roc_auc_slda = roc_auc_score(y_test, y_df)
y_pred = clf_slda.predict(X_test)
bal_acc_auc_slda = balanced_accuracy_score(y_test, y_pred)

print("\nsLDA scores with channel prime data")
print("roc_auc: ",roc_auc_slda)
print("bal_acc_auc: ",bal_acc_auc_slda)

### BT-LDA

# 19/04/2025: added from Jan's example_toeplitz_lda_simple.py:
from toeplitzlda.classification import (
    EpochsVectorizer,
    ShrinkageLinearDiscriminantAnalysis,
    ToeplitzLDA,
)

clf_ival_boundaries = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
nch = 63
# Straightforward use toeplitz lda
clf_btlda = make_pipeline(
    # EpochsVectorizer(
    #     select_ival=feature_ival,
    # ),
    ToeplitzLDA(n_channels=nch),
)
clf_btlda.fit(X_train,y_train)

y_df = clf_btlda.decision_function(X_test)
roc_auc_btlda = roc_auc_score(y_test, y_df)
y_pred = clf_btlda.predict(X_test)
bal_acc_auc_btlda = balanced_accuracy_score(y_test, y_pred)

print("\nBT LDA scores with channel prime data")
print("roc_auc: ",roc_auc_btlda)
print("bal_acc_auc: ",bal_acc_auc_btlda)

```

**Notes:** ...

**To do:** 

- (Optional) test effect of turning off channel-prime order
- (Optional) consider other ways to compute AUC
- (Optional) consider other evaluation methods?

---

### 📙 Note 3: How accuracy is measured [calibration] (2/2)

The accuracy is measured with sklearn's method [metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html):

    ```
    fpr, tpr, thresholds = metrics.roc_curve(y_test,clf.decision_function(X_test)) 
    ```
  
See comments in jupyter notebook for more details on the function

![formulas_auc](images/formulas_auc.png)

**Notes:**
- Note that the length of fpr and tpr is equal to the number of unique values obtained. This means that for one subset of the data, tpr could be of size 30 for instance, while for another subset of the same size, the fpr could be 27. 

---

## 📅 14/04/2025

### 📋 Exp 2: Use TimeSeriesSplit **[calibration]**

**Goal:** Use k-folds cv instead of a single train/test split & at the same time respect the chronological order.

**Change:** single train_test_split (no cv) --> [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) (cv)

**Results:**
I have decided not to average over folds for better visualization

![AUC_train_test_split](images/ex3_auc_new.png)
*Figure 1. AUC train_test_split*

![AUC_TimeSeriesSplit](images/ex3_auc_timeseriessplit_fold_0.png)
*Figure 2. AUC TimeSeriesSplit fold 0*

![AUC_TimeSeriesSplit](images/ex3_auc_timeseriessplit_fold_1.png)
*Figure 3. AUC TimeSeriesSplit fold 1*

![AUC_TimeSeriesSplit](images/ex3_auc_timeseriessplit_fold_2.png)
*Figure 4. AUC TimeSeriesSplit fold 2*

![AUC_TimeSeriesSplit](images/ex3_auc_timeseriessplit_fold_3.png)
*Figure 5. AUC TimeSeriesSplit fold 3*

![AUC_TimeSeriesSplit](images/ex3_auc_timeseriessplit_fold_4.png)
*Figure 6. AUC TimeSeriesSplit fold 4*

**Preprocessing/Settings:**
- test_size was kept the same for both train_test_split and TimeSeriesSplit (test size = 10%).
- same preprocessing as in 11/04/2025, with baseline correction set to `None`.
- 1080 epochs were used for this experiment (out of 3240. The other 2160 are used for online simulation)

train_test_split
```
X_train, X_test, y_train, y_test = train_test_split(calibration_stimuli, calibration_labels, test_size=0.1, shuffle=False)
```

TimeSeriesSplit
```
X = calibration_stimuli
test_size = int(0.1 * len(X)) 
timeseriescv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=test_size)
```

**Notes:**
- See (13/04/2025 Note 1), [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) for more info.
- My understanding is that TimeSeriesSplit is the same as a rolling k-fold cross-validation. "Unlike cross-validation methods, successive training sets are supersets of those that come before them." [[TimeSeriesSplit documentation]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit)
- The final fold of TimeSeriesSplit is exactly the same as the result of train_test_split. All previous folds have a worse performance, except from fold 4... Maybe it is actually not better to replace train_test_split by TimeSeriesSplit? 

---

### 📙 Note 2: sklearn's TimeSeriesSplit: parameter max_train_size 

**Topic**: Setting max_train_size to `None` looks like cv on a rolling basis. Setting max_train_size to another value, looks the same as the function in A6 that computes the AUC for different dataset sizes, using multiple samples per size 

**Change:** max_train_size = `None` vs max_train_size = 4

**Results:** 
```
TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
Fold 0:
  Train: index=[0 1]
  Test:  index=[2 3]
Fold 1:
  Train: index=[0 1 2 3]
  Test:  index=[4 5]
Fold 2:
  Train: index=[0 1 2 3 4 5]
  Test:  index=[6 7]
Fold 3:
  Train: index=[0 1 2 3 4 5 6 7]
  Test:  index=[8 9]
Fold 4:
  Train: index=[0 1 2 3 4 5 6 7 8 9]
  Test:  index=[10 11]
```

```
TimeSeriesSplit(gap=0, max_train_size=4, n_splits=5, test_size=None)
Fold 0:
  Train: index=[0 1]
  Test:  index=[2 3]
Fold 1:
  Train: index=[0 1 2 3]
  Test:  index=[4 5]
Fold 2:
  Train: index=[2 3 4 5]
  Test:  index=[6 7]
Fold 3:
  Train: index=[4 5 6 7]
  Test:  index=[8 9]
Fold 4:
  Train: index=[6 7 8 9]
  Test:  index=[10 11]
```

**To do:** 
- Maybe I can use this for the forgetting strategy with a sliding window / binary cutoff

---

## 📅 13/04/2025

### 📙 Note 1: How accuracy is measured [calibration] (1/2)

**Topic:** How the calibration data is split into a train and test set to measure LDA's accuracy

**Notes:**

- The accuracy (see 11/04/2025 Exp 1) is measured as follows:
    ```
    X_train, X_test, y_train, y_test = train_test_split(calibration_stimuli, calibration_labels, test_size=0.1, shuffle=False)
    clf = LDA().fit(X_train, y_train)

    fpr, tpr, thresholds = metrics.roc_curve(y_test,clf.decision_function(X_test)) 
    ```
- sklearn's method [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) is used to split the calibration data into a training set and test set. 
- If shuffle is `True`: the train & test set are picked randomly & not (chronologically) ordered. This is problematic in our case. The relationship between the data points x and their corresponding label y remains unchanged, however.
- If shuffle is `False`, the train/test set is selected in chronological order. This means that when having 100 data points and the train set is 80%, then the first 80 data points are selected. It's good that the chronological order remains, but is it reliable to base off the accuracy on only one sample of [0-80]? We could also take [10-90], or [20-100]. Even better would be cross-validation, but we should respect the chronological order. Nevertheless, a K-fold cross validation might still be better. This will be revisited in 25/04/2025.

**To do:**
- (Optional) Test if/how taking different sample intervals for the train_test_split would affect the accuracy.
- Use sklearn's solution for time series data:  [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) 

---


## 📅  11/04/2025

### 🔧  MDF 1: Turn off baseline correction

**Modification:** Turn off baseline correction in the preprocessing steps.

### 📋  Exp 1: Effect of baseline correction on LDA [calibration]

**Goal:** Test the effect of baseline correction on LDA in calibration

**Change:** changed basline [-0.2, 0 s] --> `None`

**Results:** Accuracy LDA: baseline - 0.799 %, no baseline = 0.820

Important note on results: the AUC score here is not really reliable... It depends on the chosen interval of trials.

![AUC1](images/ex3_auc_new.png)
*Figure 1. AUC without baseline-correction*

![AUC2](images/ex3_auc_old.png)
*Figure 2. AUC with baseline-correction*

**Preprocessing/Settings:**    
- Preprocessing:
    - Bandpass-filtering = (0.5, 16 Hz)
    - `raw.filter(*filter_band, method="iir")`
    - Baseline interval = [-0.2, 0] or `None` 
    - Sampling rate 1000 Hz --> down sampled to 100 Hz
    - Outlier rejection: None 

- Epochs:
    - tmin = -0.2 s 
    - tmax = 1.0 s 
    - 63 EEG channels x 4 time intervals = 252 features
    - 1080 epochs used (out of 3240 epochs in total; see notes on dataset)

- Evaluation (how accuracy was measured):
```
X_train, X_test, y_train, y_test = train_test_split(calibration_stimuli, calibration_labels, test_size=0.1, shuffle=False)
clf = LDA().fit(X_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_test,clf.decision_function(X_test)) # Compute signed distance of stimulus to decision boundary
auc_fig = metrics.RocCurveDisplay(fpr=fpr, tpr = tpr)
auc_fig.plot()
plt.plot([0, 1],[0,1], '--')
plt.legend(['ROC (area = %0.3f)' % metrics.auc(fpr, tpr), 'area = 0.5'], loc="lower right")
plt.title("AUC-ROC Curve of the LDA classifier")
plt.show()
```



**Notes:** 
- In the original file, there was no baseline parameter given. When epochs are extracted with ` epoch = mne.Epochs(...)`, the epochs have a standard baseline correction of [-0.2, 0 s] (Default of mne.Epochs)
- I added `baseline = None`, so now that line is changed to `    epoch = mne.Epochs(..., baseline=None)` 
- This should be advantegous for BT-LDA, but it also appeared better for normal LDA (see results).
- The AUC score here is not really reliable... It depends on the chosen interval of trials.

**To do:**
- Check this effect on Block-Toeplitz LDA

---
