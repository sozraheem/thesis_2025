This folder contains the performance scores of all classifiers that have been compared in my offline analysis. The description of each version can be found in the thesis.

- aphasia_sldadata.csv: the Adaptive CC sLDA (of the original experiment) (2nd baseline)

- ccdata.csv: the Adaptive CC BT-LDA with the optimized UC-pair of even patients, used for odd patients, and the optimized UC-pair for odd patients, used for even patients.

- staticdata.csv: the Static Fixed BT-LDA

- transferdata.csv: the Transfer Fixed BT-LDA (1st baseline)

- window4data.csv: the Adaptive Window BT-LDA modified: instead of using a predefined moving window size, the amount of epochs of the previous session is used for every new session. This version is not included in the comparison in my thesis.

- window5data.csv: the Adaptive Window BT-LDA described in the thesis (using a moving window size of 3600 epochs)