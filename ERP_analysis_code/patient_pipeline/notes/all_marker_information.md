# Marker information of all patients

## All patients 1-10 have the following markers:
```
[  101   102   103   104   105   106   
   111   112   113   114   115   116   
   200   201   202   203   204   205   ]
```

## Extra markers:
```
#### Patient 1: [206   207                     238                           255 10001 99999]

#### Patient 2: [206   207                                                   255 10001 99999]

#### Patient 3: [            210   211   214         243   247   250   251   255 10001 99999]

#### Patient 4: [            210   211   214         243   247   250   251   255 10001 99999]

#### Patient 5: [            210   211   214               247         251   255 10001 99999]

#### Patient 6: [            210   211   214         243   247         251   255 10001 99999]

#### Patient 7: [            210   211                                       255 10001 99999]

#### Patient 8: [            210   211   214         243   247         251   255 10001 99999]

#### Patient 9: [            210   211                                 251   255 10001 99999]

#### Patient 10: [           210   211                                       255 10001 99999]
```

## Special remarks

#### Patient 1
Looking at all the filenames in the `p1_marker_info.log` in "Searching for the uncommon markers: [  206   207   238 10001]":

- In all files, $S206$ appeared as the 2nd marker, after New Segment and before $S20*$ (e.g. $S204$)
- In all files, $S207$ also appeared as the 2nd marker, after New Segment and before $S20*$ (e.g. $S205$)
- DC Correction ($S10001$) took place in P1_S9_Block1_Run6 at Mk331=DC Correction,,215041,1,0. This was after a trial with target word $S112$ and before marker $S205$ (i.e., before a trial with target word $S116$)
- $S238$ appeared in P1_S13_Block2_Run5 as the 2nd marker, after New Segment and before $S204$

#### Patient 2
For patient 2, session 13: auditoryAphasia_6D_350_Block3_Run5.vmrk contains only
```
Mk1=New Segment,,1,1,0,19200101165957306035
Mk2=Stimulus,S200,6290,1,0
```
And this messed up the log file of common markers across all runs in session 13. Therefore, I have decided to move this one into a new folder in the same directory called `2025_bin`. I also moved the corresponding .vmrk and .eeg file (which was relatively very small, ~1000 KB) into `2025_bin`. I then ran the log function that computed common and odd markers on patient 2 again. The new log file is called `p2_marker_info_cleaned`. Note that this new file does not contain P2_S13/anonymized/auditoryAphasia_6D_350_Block3_Run5.vmrk anymore.

Analyzing the `p2_marker_info_cleaned.log` file:
"Searching for the uncommon markers: [  206   207 10001]":

- In all 4 files, $S206$ appeared as the 2nd marker, after New Segment and before $S20*$ ($S200$ or $S202$)
- $S207$ appeared in one file: as the 2nd marker, after New Segment and before $S201$.

Note that patient 2 had a SOA of 500 ms in some runs... I have to look into this later. I saw that in Session 11, SOA 350 became SOA 500 in the middle of the session. In S12, the SOA was 500 and in S13, the SOA was 350.

#### Patient 3
Looking at all the filenames in the `p3_marker_info.log` in "Searching for the uncommon markers: [  214   243   247   250   251 10001]":

- In all 3 files, marker $S214$ appears as the 2nd marker, after New Segment and before $S210$. The marker after $S210$ is a $S20*$ marker (with * in 0-5)
- For $S243$ I could not see a pattern. In one file is appeared after $S211$ (which I think is the dynamic stopping marker) and in another file it appeared before $S211$. 
- $S247$ appeared in 14 files. It almost always appeared before $S211$ (Which was often followed by a single cue marker, e.g. $S101$, and then $S210$), but sometimes $S247$ appeared after $S211$, after which ~3 cue markers were found and then the run end marker $S255$.
- $S250$ appeared in 2 files: as the 2nd marker, after New Segment and before $S210$.
- $S251$ followed a similar pattern as $S247$
- $S10001$ appeared in 3 places (I have not looked at this exactly)

#### Patient 4
Looking in `p4_marker_info.log`. For patient 4, session 9: auditoryAphasia_6D_350_Block3_Run6.vmrk contains only the following unique markers: [101 102 103 104 105 106 111 113 200 202 210 211], which is strange. There are 104 markers in total, and it seems that only trial 1 + the first iteration of trial 2 is there. The corresponding .eeg file is ~6500 KB, while other .eeg files are ~21000 KB. 

And this messed up the log file of common markers across all runs in session 19. Therefore, I have decided to move this one into a new folder in the same directory called `2025_bin`. I also moved the corresponding .vmrk and .eeg file into `2025_bin`. I then ran the log function that computed common and odd markers on patient 4 again. The new log file is called `p4_marker_info_cleaned`. Note that this new file does not contain P4_S19/anonymized/auditoryAphasia_6D_350_Block3_Run6.vmrk anymore.