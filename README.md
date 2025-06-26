# BSc_Thesis
This repository contains the files that I have been working on for my BSc Thesis

## Contents
`bsc_thesis_soz`: my bachelor's thesis. This is the updated version. Appendix A was missing in the first version and corrected.
    
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
After the thesis submission deadline, this repository was cleaned up and reorganized. Additional ackowledgements/credits were added where appropriate. Upon review, I also discovered a few minor mistakes in the originally submitted thesis. I am currently correcting these and aim to upload the updated version soon.

