# A7 Experiments/Modifications Log

## 11/04/2025

For comparisons with the 'old' file, I refer to _old_assignment_07_dynamic_stopping_solutions_modified_Copy1_

### Mod 1: Turn off baseline correction
- **Change:** Set baseline correction to `None`
- **Results:**

- **Notes:**    
    - In the original file, there was no baseline parameter given. When epochs are extracted with ` epoch = mne.Epochs(raw, events=evs, event_id=event_id, decim=decimate,
                      proj=False, tmax=1)`, the epochs have a standard baseline correction of [-0.2, 0 s] (Default of mne.Epochs)
    - I changed it to `    epoch = mne.Epochs(raw, events=evs, event_id=event_id, decim=decimate,
                       proj=False, tmax=1, baseline=None)` where I added `baseline = None`
    - This should be advantegous for BT-LDA, but it also appeared better for normal LDA (see results).
- **To do:**
    - Check this effect on Block-Toeplitz LDA

---

### Mod 2: