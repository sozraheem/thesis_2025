# BSc_Thesis
This repository contains the files that I have been working on for my BSc Thesis

## Contents
`bsc_thesis_soz_raheem_updated`: my updated bachelor's thesis. The applied modifications are listed in **Note to assessors** below in this file.
    
`ERP_analysis_code` includes the following files:
- `experiments`: contains all experiments I have conducted on Block-Toeplitz LDA [1] (on data from [2]) prior to applying it on the real dataset
- `patient_pipeline`: contains all code I have used for my thesis, applied to the real dataset [3]
- `environment.yml`
- `requirements.txt`

## Environment setup
I used Conda to create my environment & install packages. The full Conda environment can be recreated with

```conda env create -f environment.yml```

Alternatively, a pip environment can be created with

`pip install -r requirements.txt`


## Dataset
This code is only runnable on the anonymized data files. The dataset [3] is currently not publicly available (June, 2025).

## Sources
- [1] J. Sosulski and M. Tangermann, “Introducing block-Toeplitz covariance matrices to remaster linear discriminant analysis for event-related potential brain–computer interfaces,” J. Neural Eng., vol. 19, no. 6, p. 066001, Nov. 2022, doi: 10.1088/1741-2552/ac9c98.
- [2] This thesis contains code that is adapted and originally obtained from the BCI Bachelor Course at Radboud University, developed by Michael Tangermann and Jordy Thielen.
- [3] M. Musso et al., “Aphasia recovery by language training using a brain–computer interface: a proof-of-concept study,” Brain Communications, vol. 4, no. 1, p. fcac008, Feb. 2022, doi: 10.1093/braincomms/fcac008.
- For assistance with the implementation of Block-Toeplitz LDA, I used this [repository](https://github.com/thijor/eeg_tutorial_erp) and this [repository](https://github.com/jsosulski/toeplitzlda). To anonymize the dataset, I used this [repository](https://github.com/simonkojima/anonymize-bv).

## Note to assessors
After the thesis submission deadline, this repository was cleaned up and reorganized. Additional ackowledgements/credits were added where appropriate. 

I have updated my thesis and applied the following modifications.
- Corrected spelling mistakes and missing citations where required
- _Background_: formula for the mean-centered data (Equation (2.3)) was wrong and corrected; added LDA's assumptions; renamed bias $w_0$ to $b$; shortened some parts
- _Methods_: improved explanation of Section 3.1.2 (experimental setup), corrected all figures; corrected Section 3.1.4 (Classifier in the protocol); corrected Section 3.4 (Proposed Adaptation Strategies), especially Section 3.4.2 now has the correct mathematics on how the parameters are computed (which was wrong in the previous version).
- _Results_: improved plots in Section 4.2.2 (Grand Average); moved the results of _Transfer Fixed_ to the Appendix; added Figure 4.7
- _Discussion_: added Section 5.3.1; small improvements of some paragraphs
- _Appendix_: added Appendix A.1 (Experimental Conditions), which was missing in the previous version; added Appendix B, C, and D.
