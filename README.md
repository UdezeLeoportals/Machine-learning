# Machine-learning
Classification-based modeling with ensemble ML and resampling techniques such as SMOTE and RandomUnderSampler using Python

The first file ensemble_credit_rerun1 consists of a stacking ensemble model which uses 5 base classifiers including k-NN, Random Forest, Gaussian Naive Bayes, Decision Tree (CART) AND SVM Classifier; and uses Logistic Regression as the meta classifier. It also implemented the SMOTE oversampling and RandomUnderSampler undersampling technique on the imbalanced datasets. The program generates 5 metrics including accuracy, precision, recall, f1-score and ROC-AUC. This program tries to detect credict card fraud detection by classifying transactions into two binary classes - fraudulent and legitimate. It was tested on 3 publicly available datasets:
https://www.kaggle.com/mlg-ulb/creditcardfraud ,
https://github.com/gksj7/creditcardcsvpresent/blob/main/creditcardcsvpresent.csv  ,
http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

The scond file credit_fraud2 compares the performance of 3 algorithms: XGBoost, Random Forest and TensorFlow DNN. It was tested on the Kaggle Dataset above and uses the ADASYN oversampling technique and RandomUnderSampler for balancing of dataset.

The third file nelson_regression performs stacking ensemble with 3 base learners: SVM Regressor, Decision Tree (CART) and k-NN Regressor using Linear Regression and the meta-learner. It was developed for the prediction of a continuous value of a chemical compound in  computational chemistry. The performance metrics used include: root_mean_sqaured_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error and r2 score.
