# A7 Experiments/Modifications Log

To do: 
- add icons to visually scan faster through this file
- assign IDs to experiments?

Experiment ideas:
- BT-LDA with different time intervals
- Test the effect of channel-prime order on 19/04/2025_Exp_1
- Test other evaluation metrics in ex 3 (calibration)
- See to do lists of previous experiments & notes


Overview
- **21/04/2025_Exp_2:** Implement first draft adaptive LDA on online data: sliding window with different step sizes
- **21/04/2025_Exp_1:** Compare static LDA vs SLDA vs BT-LDA on online data (using AUC-ROC curves, ex.4.1)
- **19/04/2025_Exp_1:** Compare LDA vs SLDA vs BT-LDA on calibration data using Jan's evaluation method
- **18/04/2025_Note_1:** How accuracy is measured in ex. 3 (calibration) 2/2
- **14/04/2025_Exp_1:** Use TimeSeriesSplit in ex. 3 (calibration)
- **14/04/2025_Note_1:** sklearn's TimeSeriesSplit: parameter max_train_size
- **13/04/2025_Note_1:** How accuracy is measured in ex. 3 (calibration) 1/2
- **11/04/2025_Exp_1:** Turn off baseline correction

## 📅 New date template

### 📙 Exp 1: [Title of experiment]

**Goal**: ...

**Change:** ...

**Results:** ...

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---

## 📅21/04/2025

### 📙 Exp 2: Implement first draft adaptive LDA on online data: sliding window with different step sizes

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

### 📙 Exp 1: Compare LDA vs SLDA vs BT-LDA on online data (using AUC-ROC curves, ex.4.1)

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

### 📙 Exp 1b: Compare LDA vs SLDA vs BT-LDA on calibration data using ex.3 evaluation method

**Goal**: Compare AUC scores of LDA, sLDA and BT-LDA on the calibration data, using the evaluation method of assignment 7, ex. 3

**Results:** ... (Add plots)

**Preprocessing/Settings:** Same as 19/04/2025_Exp_1a

**Notes:** ...

**To do:** ...

---

## 📅 19/04/2025

### 📙 Exp 1a: Compare LDA vs SLDA vs BT-LDA on calibration data using Jan's evaluation method

**Goal**: Compare AUC scores of LDA, sLDA and BT-LDA on the calibration data. The evaluation method of Jan's `example_toeplitz_lda_simply.py` script was used.

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
- Consider other ways to compute AUC
- Consider other evaluation methods?

---

### 📙 Note 1: How accuracy is measured in Ex. 3 (Calibration) Part 2/2

The accuracy is measured with sklearn's method [metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html):

    ```
    fpr, tpr, thresholds = metrics.roc_curve(y_test,clf.decision_function(X_test)) 
    ```
  
See comments in jupyter notebook for more details on the function

![formulas_auc](images/formulas_auc.png)

---

## 📅 14/04/2025

### 📙 Exp 1: Use Skicit's TimeSeriesSplit in Exercise 3 (Calibration) 

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
- I have decided to stick with the train_test_split

---

### 📙 Note 1: sklearn's TimeSeriesSplit: parameter max_train_size

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
- (Optional) implement this in A6 and see if is exactly the same as my function

---

## 📅 13/04/2025

### 📙 Note 1: How accuracy is measured in Ex. 3 (Calibration) Part 1/2

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
- If shuffle is `False`, the train/test set is selected in chronological order. This means that when having 100 data points and the train set is 80%, then the first 80 data points are selected. It's good that the chronological order remains, but is it reliable to base off the accuracy on only one sample of [0-80]? We could also take [10-90], or [20-100]. Even better would be cross-validation, but we should respect the chronological order.

**To do:**
- (Optional) Test if/how taking different sample intervals for the train_test_split would affect the accuracy.
- Use sklearn's solution for time series data:  [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) --> See 14/04/2025_exp_1
- Look at assignment 6 how this is done there. It probably needs to change too, as I think the chronological order is not respected there either.

---


## 📅  11/04/2025

### 📙  Exp 1: Turn off baseline correction in calibration

**Goal:** Test effect of baseline correction on LDA in calibration

**Change:** changed basline [-0.2, 0 s] --> `None`

**Results:** Accuracy LDA: baseline - 0.799 %, no baseline = 0.820

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
- For comparisons with the 'old' file, I refer to _old_assignment_07_dynamic_stopping_solutions_modified_Copy1 
- In the original file, there was no baseline parameter given. When epochs are extracted with ` epoch = mne.Epochs(...)`, the epochs have a standard baseline correction of [-0.2, 0 s] (Default of mne.Epochs)
- I added `baseline = None`, so now that line is changed to `    epoch = mne.Epochs(..., baseline=None)` 
- This should be advantegous for BT-LDA, but it also appeared better for normal LDA (see results).

**To do:**
- Check this effect on Block-Toeplitz LDA

---
