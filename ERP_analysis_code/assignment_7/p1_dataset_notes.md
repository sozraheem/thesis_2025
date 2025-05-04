# Notes on data of patient 1

### Overview
1. General session information
2. Experimental setup
3. Dataset
4. Preprocessing

## General session information
There are 18 sessions. Session 1, 2, and 18 are offline sessions. Sessions 3-17 are online. 
See `p1_data_structure.txt` (in assignment_7/other) for an overview of all sessions.
In all online sessions (3-17) the conditions are the same: 6D 350

Updating should be implemented in the online sessions.
In the auditory aphasia therapy the classifier was trained on session n-1 and taken into session n, where then updating took place.


## Experimental setup: auditory aphasia paradigm
The visualization below was created for the data of assignment 7
**To do:** change the content to that of patient 1 instead

![dataset](images/A7_dataset.png)


### Summary (bottom-up)
- 6 words/stimuli per iteration
    - one is the target **t** and five are non-targets **nt**.
- $\le$ 15 iterations form a single trial which means $\le$ 90 stimuli per trial

    - within a trial, the target word is always the same.
- 6 trials form a single run: $\le$ 6\*90 = 540 stimuli per run
- 6 runs form a block: $\le$ 6\*540 = 3240 stimuli in total

With no dynamic stopping a trial contains 15 iterations. However, with dynamic stopping the number of iterations differ (and therefore, the number of epochs/stimuli differs too). More information is to be added.

## Dataset
The dataset consists of 18 sessions, of which session 1, 2, and 18 are offline sessions. Each session has a varying number of blocks. Each block contains a maximum of 6 runs. 

Per run an `.eeg`, `.vhdr`, and `.vmrk` file is provided. See the jupyter notebook `p1_notebook.ipynb` for a more detailed description of the content of these files. (Maybe I am eventually going to transform the `p1_notebook.ipynb` into a Python script and copy all text from there into this file).

See `p1_dataset_structure.txt` for an overview of how the data is structured.

### Markers
In each `.vmrk` file the event marker information is found. Every stimulus has an `event_id`, which is on of the following:

- event_ids $[S101, S102, ..., S106]$ are `[Word_1/Nontarget, Word_2/Nontarget, ..., Word_6/Nontarget]` respectively
- event_ids $[S111, S112, ..., S116]$ are `[Word_1/Target, Word_2/Target, ..., Word_6/Target]` respectively
- event_ids $[S200, S201, S202, S203, S204, S205]$ are the starting markers of a single trial. 
    - $S200$ means that a trial has just started with `Word_1/Target` as the target word in the whole trial
    - $S201$ means that a trial has just started with `Word_2/Target` as the target word in the whole trial
    - ...
    - $S205$ means that a trial has just started with `Word_6/Target` as the target word in the whole trial
- event_id $S255$ indicates the end of the run    

In short: the first digit indicates whether it is the start/end of a trial (2) (note that this is not a played word) or just a stimulus/word that is played (1). For all played stimuli, the second digit indicates whether it is a target word (1) or nontarget word (0) and third digit indicates the word id

## Preprocessing (flowchart?)

The following preprocessing is copied from A7_notebook. Do I stick with this or switch to David's preprocessing settings?

- Preprocessing:
    - only EEG files are selected
    - Bandpass-filtering = (0.5, 16 Hz)
    - `raw.filter(*filter_band, method="iir")`
    - Baseline interval = [-0.2, 0] --> `None` 
    - Sampling rate 1000 Hz --> down sampled to 100 Hz
    - Outlier rejection: `None `

- Epochs:
    - tmin = -0.2 s 
    - tmax = 1.0 s 
    - in offline sessions: 63 EEG channels 
    - in online sessions: 31 EEG channels
    - averaged time intervals: 4 (but this may change later on)


Eventually the final preprocessing will be added here.

## Source
To be added. 

Paper by Musso et al. 

Data preprocessing & description obtained from assignment 7 of the BCI course.

.

.

.

.

.

.

.


## Explanation to add later in the experimental setup
Dataset description
In every trial the patient has to focus on a single word from the set of 6 monosyllabic words, played on 6 speakers. The model has to decode the target word, i.e., which word the patient is attending to in that trial. To gather enough data for this task, the sequence of 6 words, a so-called *iteration*, is repeated 15 times. That means that there are 15 iterations in a single trial, adding up to 90 words/stimuli per trial. 

So, an iteration consists of 6 words: 5 non-targets and 1 target. Among iterations (within a trial) the target word is the same, but the order of words differ. 
15 iterations form a single trial. Per trial, the decoding model decides what the target word is. 6 trials form a single run. After each run, the patient can take a break. A single session/block consists of 6 runs.

.

## Copied from the BCI course's assignment 7:


#### Experimental setup:
In every trial the patient/user (from now on 'participant') has to focus on a single word from the set of 6 words. Each word/stimulus is played once per iteration. This means that an iteration consists of 6 words/stimuli, of which one is the target word. A sequence of 15 iterations, adding up to 90 stimuli, form one complete trial. The goal of a single trial is to determine what the target word is. This means that each iteration in a single trial shares the same target word. However, the order in which words are presented between iterations in a single trial may differ. To summarise, each trial has a single target word. Each trial consists of 15 iterations in which all 6 words are played once.

#### General EEG data information:
The data provided contains three file types: `.vhdr`, `.vmrk`, `.eeg`.  In the each `.vhdr` file you will find information about all recorded channels. Five of the channels listed are non-EEG channels:
* The EMG channel records an electromyogram. This is muscle activity.
* The GSR channel records the galvanic skin response. This is sweat gland activity which is indicative of stress levels and excitation.
* The Respi channel records respiration activity.
* The Pulse channel records the heart pulses by shining a red light on the finger and recording how much of it is reflected back.
* The Optic channel is an optical sensor focused on a portion of the screen that flashes every time an event happens in order to detect potential interference/delay between the time point the computer issues a stimulus and the time the stimulus is actually presented on the screen to the user.

Some of these channels can be used to remove artefacts from the EEG signal and better phase-locking the signals. For our experiment though they aren't relevant.

In the `.vhdr` files you will also find the resolution ( $\mu V$ steps) of each channel and the impedance (kOhm) of all channels, a higher impedance results in a higher noise level.

In each `.vmrk` you will find the event marker information. Each stimulus/event has an `event_id`. In our case, non-target words have event_ids $[101, 102, ..., 106]$. We have 6 different words. In every trial exactly 1 of them is in the target role. The target words are indicated by $[111,112, ..., 116]$. So, the last digit indicates the word id, the second digit indicates whether a stimulus is a target $(1)$ or non-target $(0)$.

Lastly, each `.eeg` file contains the recorded signals encoded in binary values and stored as integers. Increasing the value by 1 corresponds to a step value of 0.1 $\mu V$, which is the resolution denoted in the header (`.vhdr`) file.
