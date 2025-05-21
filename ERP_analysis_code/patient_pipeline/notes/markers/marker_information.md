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
Looking at all the filenames in the `p1_marker_log` in "Searching for the uncommon markers: [  206   207   238 10001]":

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
And this messed up the log file of common markers across all runs in session 13. Therefore, I have decided to move this one into a new folder in the same directory called `2025_bin` and then run the log function that computed common and odd markers on patient 2 again. The new log file is called `p2_marker_info_corrected`. Note that this new file does not contain P2_S13/anonymized/auditoryAphasia_6D_350_Block3_Run5.vmrk anymore.