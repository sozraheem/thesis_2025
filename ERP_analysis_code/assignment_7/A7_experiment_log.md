# A7 Experiments/Modifications Log

To do: 
- add icons to visually scan faster through this file
- assign IDs to experiments?

Overview
- 14/04/2025_Exp_1: Use TimeSeriesSplit in ex. 3 (calibration)
- 14/04/2025_Note_1: sklearn's TimeSeriesSplit: parameter max_train_size
- 13/04/2025_Note_1: How accuracy is measured in ex. 3 (calibration)
- 11/04/2025_Exp_1: Turn off baseline correction

## 📅 New date template

### 📙 Exp 1: [Title of experiment]

**Goal**: ...

**Change:** ...

**Results:** ...

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...


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
- The final fold of TimeSeriesSplit is exactly the same as the result of train_test_split. All previous folds have a worse performance... Maybe it is actually not better to replace train_test_split by TimeSeriesSplit? 

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

### 📙 Note 1: How accuracy is measured in Ex. 3 (Calibration)

**Topic:** How LDA's accuracy currently is calculated on calibration data

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
- Use sklearn's solution for time series data:  [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) --> See 13/04/2025_exp_1
- Look at assignment 6 how this is done there. It probably needs to change too, as I think the chronological order is not respected there either.

---


## 📅  11/04/2025

### 📙  Exp 1: Turn off baseline correction

**Goal:** Test effect of baseline correction on LDA

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
    - 3240 epochs in total (see notes on dataset)

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
