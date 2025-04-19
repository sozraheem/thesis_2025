# Debugging steps when implementing ToeplitzLDA


## 19/04/2025

I try to implement the BT-LDA implementation of Jan's example script `example_toeplitz_lda_simple.py` in A7, but I encountered the following error:
```
AttributeError                            Traceback (most recent call last)
Cell In[15], line 61
     54 # Straightforward use toeplitz lda
     55 clf_btlda_test = make_pipeline(
     56     EpochsVectorizer(
     57         select_ival=feature_ival,
     58     ),
     59     ToeplitzLDA(n_channels=nch),
     60 )
---> 61 clf_btlda_test.fit(X_train,y_train)
     63 y_df = clf_btlda_test.decision_function(X_test)
     64 roc_auc_btlda_test = roc_auc_score(y_test, y_df)

File c:\Users\Soz\anaconda3\envs\thesis\Lib\site-packages\sklearn\base.py:1389, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
   1382     estimator._validate_params()
   1384 with config_context(
   1385     skip_parameter_validation=(
   1386         prefer_skip_nested_validation or global_skip_validation
   1387     )
   1388 ):
-> 1389     return fit_method(estimator, *args, **kwargs)

File c:\Users\Soz\anaconda3\envs\thesis\Lib\site-packages\sklearn\pipeline.py:654, in Pipeline.fit(self, X, y, **params)
    647     raise ValueError(
...
---> 78     e.crop(tmin=self.select_ival[0], tmax=self.select_ival[1])
     79     self.times_ = e.times
     80     X = e.get_data() * self.scaling

AttributeError: 'numpy.ndarray' object has no attribute 'crop'
```

For Jan's example scripts which are runnable right now, I use the conda environment `toeplitz_venv` which can be found in the Toeplitz repository. For A7, I use the conda environment `thesis` which can be found in this repository (see environment.yml and requirements.txt).

When looking up the \Lib\site-packages\sklearn\base of both environments I see differences. I checked the environment.yml files of both and saw that `toeplitz_venv` uses  scikit-learn==1.3.2, while `thesis` uses       - scikit-learn==1.6.1
This might cause problems later.

To make it work I commented out 
```
     56     EpochsVectorizer(
     57         select_ival=feature_ival,
     58     ),
```
because we have already epoched our data, so it does not make sense to include this in the pipeline. The issue was probably caused by the np array that was passed, while an instance of mne.Epochs was expected.

Now it runs and it works.