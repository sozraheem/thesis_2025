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

For patient 2, in session 13: auditoryAphasia_6D_350_Block3_Run5.vmrk contains only
```
Mk1=New Segment,,1,1,0,19200101165957306035
Mk2=Stimulus,S200,6290,1,0
```
And this messed up the log file of common markers across all runs in session 13. Therefore, I think I should remove this one.