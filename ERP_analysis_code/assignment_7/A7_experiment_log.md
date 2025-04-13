# A7 Experiments/Modifications Log

## 11/04/2025
For comparisons with the 'old' file, I refer to _old_assignment_07_dynamic_stopping_solutions_modified_Copy1_



### Exp 1: Turn off baseline correction

**Goal:** Test effect of baseline correction on LDA

**Change:** changed basline [-0.2, 0 s] --> `None`

**Results:**
- Accuracy LDA: baseline - 0.799 %, no baseline = 0.820

![AUC1](images/ex3_auc_new.png)

![AUC2](images/ex3_auc_old.png)

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


**Notes:**    
- In the original file, there was no baseline parameter given. When epochs are extracted with ` epoch = mne.Epochs(...)`, the epochs have a standard baseline correction of [-0.2, 0 s] (Default of mne.Epochs)
- I added `baseline = None`, so now that line is changed to `    epoch = mne.Epochs(..., baseline=None)` 
- This should be advantegous for BT-LDA, but it also appeared better for normal LDA (see results).

**To do:**
- Check this effect on Block-Toeplitz LDA

---

### Mod 2:
**Change:** ...

**Results:** ...

**Preprocessing/Settings:** ...

**Notes:** ...

**To do:** ...

---
