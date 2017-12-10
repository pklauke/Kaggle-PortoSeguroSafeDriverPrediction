# Kaggle-PortoSeguroSafeDriverPrediction

This repository contains my solution for the Porto Seguro Safe Driver Prediction competition: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction. This solution ranked 138 of 5170.

The goal of this competition was to find out whether a driver will make a claim or not. Due to the fact that the data was anonymized feature engineering was difficult. Therefore my solution focuses mainly on emsembling a variety of models using weighted blending.

The majority of models were trained in the notebook <a href = "https://github.com/pklauke/Kaggle-PortoSeguroSafeDriverPrediction/blob/master/Predictor.ipynb"> Predictor </a>. This includes the gradient Boosted Decision Trees <a href = "http://lightgbm.readthedocs.io/en/latest/">LightGBM</a>, <a href= "http://xgboost.readthedocs.io/en/latest/">XGBoost</a>, <a href = "https://catboost.yandex">CatBoost</a> and the Regularized Greedy Forest <a href = "https://github.com/fukatani/rgf_python">rgf_python</a>. For more variety 3 modified versions of public kernels from other Kaggle users were used aswell. These kernels used LightGBM, XGBoost, rgf_python and keras. 

In addition the Field-Aware Factorization Machine <a href = "https://github.com/guestwalk/libffm">LibFFM</a> was trained in the notebook <a href = "https://github.com/pklauke/Kaggle-PortoSeguroSafeDriverPrediction/blob/master/LibFFM%20Predictor.ipynb"> LibFFM Predictor </a>. The used library is a <a href = "https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43741"> modified LibFFM </a> version that supports early stopping with patience.

All models were trained using mostly 5-fold cross validation multiple times. Each model was trained each time on different cross validation random seeds. The predictions of all runs of a model were averaged. The averaged predictions were blended in the last step.

The models were ensembled in the notebook <a href = "https://github.com/pklauke/Kaggle-PortoSeguroSafeDriverPrediction/blob/master/Ensembling.ipynb"> Ensembling </a>. The ensembling method used was weighted blending. A self-written optimization algorithm was used to optimize the weights using the out-of-fold predictions. This algorithm decided not to use XGBoost and CatBoost for the final submission.

