# Notes on data of patient 1

### Overview
1. General session information
2. Experimental setup
3. Dataset (technical info)
4. Markers
5. Preprocessing (to be updated / decided still)

## 1. General session information
For patient 1, there are 18 sessions. Session 1, 2, and 18 are offline sessions. Sessions 3-17 are online. 
In all online sessions (3-17) the conditions are the same: 6D 350.
For the information of all other patients, I refer to patient_pipeline/notes/auditory_aphasia_data_analysis/session_conditions.md

In the auditory aphasia therapy the classifier was trained on session n-1 and taken into session n, where then updating took place [1].


## 2. Experimental setup: auditory aphasia paradigm

In every trial the patient has to focus on a single word from the set of 6 monosyllabic words, played on 6 speakers. The model has to decode the target word, i.e., which word the patient is attending to in that trial. To gather enough data for this task, the sequence of 6 words, a so-called *iteration*, is repeated 15 times. That means that there are 15 iterations in a single trial, adding up to 90 words/stimuli per trial. 

So, an iteration consists of 6 words: 5 non-targets and 1 target. Among iterations (within a trial) the target word is the same, but the order of words differ. 
15 iterations form a single trial. Per trial, the decoding model decides what the target word is. 6 trials form a single run. After each run, the patient can take a break. .

### Summary (bottom-up)
- 6 words/stimuli per iteration
    - one is the target **t** and five are non-targets **nt**.
- $\le$ 15 iterations form a single trial which means $\le$ 90 stimuli per trial

    - within a trial, the target word is always the same.
- 6 trials form a single run: $\le$ 6\*90 = 540 stimuli per run
- 6 runs form a block: $\le$ 6\*540 = 3240 stimuli in total

With no dynamic stopping a trial contains 15 iterations. However, with dynamic stopping the number of iterations differ (and therefore, the number of epochs/stimuli differs too). More information is to be added.

## 3. Dataset (technical info)
The dataset of patient 1 consists of 18 sessions, of which session 1, 2, and 18 are offline sessions. Each session has a varying number of blocks. Each block contains a maximum of 6 runs. 

For every run a `.eeg`, `.vhdr` and `.vmrk` file is provided. 

The `.vhdr` file of P1 (patient 1), S1 (session 1), Block1, Run1 contains the following information:
- number of channels: 69 (63 are EEG)
- sampling rate: 1000 Hz
- frequency range: [0-250 Hz]

Note that in the online sessions (session 3-17), the number of EEG channels is not 63, but 31.

The following channels are non-EEG channels:
* The EMG channel records an electromyogram. This is muscle activity.
* The GSR channel records the galvanic skin response. This is sweat gland activity which is indicative of stress levels and excitation.
* The Respi channel records respiration activity.
* The Pulse channel records the heart pulses by shining a red light on the finger and recording how much of it is reflected back.
* The Optic channel is an optical sensor focused on a portion of the screen that flashes every time an event happens in order to detect potential interference/delay between the time point the computer issues a stimulus and the time the stimulus is actually presented on the screen to the user.

In the `.vhdr` files there is also the resolution ( $\mu V$ steps) of each channel.

In the `.vmrk` file marker information is found about the events.


## 4. Markers
In each `.vmrk` file the event marker information is found. Every stimulus has an `event_id`, which is one of the following:
```
[  101   102   103   104   105   106   111   112   113   114   115   116
   200   201   202   203   204   205   206   207   238   255 10001 99999]
```

- event_ids $[S101, S102, ..., S106]$ are `[Word_1/Nontarget, Word_2/Nontarget, ..., Word_6/Nontarget]` respectively
- event_ids $[S111, S112, ..., S116]$ are `[Word_1/Target, Word_2/Target, ..., Word_6/Target]` respectively
- event_ids $[S200, S201, S202, S203, S204, S205]$ are the starting markers of a single trial. 
    - $S200$ means that a trial has just started with `Word_1/Target` as the target word in the whole trial
    - $S201$ means that a trial has just started with `Word_2/Target` as the target word in the whole trial
    - ...
    - $S205$ means that a trial has just started with `Word_6/Target` as the target word in the whole trial
- event_id $S255$ indicates the end of the run    
- event_ids $S206, S207, S238, S10001, S99999$ are not clear yet

In short: the first digit indicates whether it is the start/end of a trial (2) (note that this is not a played word) or just a stimulus/word that is played (1). For all played stimuli, the second digit indicates whether it is a target word (1) or nontarget word (0) and third digit indicates the word id

## 5. Preprocessing 

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
    - in offline sessions: 63 EEG channels --> discard extra channels --> 31 EEG
    - in online sessions: 31 EEG channels
    - averaged time intervals: [0.1,0.15,...,0.75,0.8] in seconds


## Sources

Some code snippets used in this repository were adapted from the BCI Bachelor Course at Radboud University, developed by Michael Tangermann and Jordy Thielen. I gratefully acknowledge the use of these resources in the implementation of my work.

[1] M. Musso et al., “Aphasia recovery by language training using a brain–computer interface: a proof-of-concept study,” Brain Communications, vol. 4, no. 1, p. fcac008, Feb. 2022, doi: 10.1093/braincomms/fcac008.

