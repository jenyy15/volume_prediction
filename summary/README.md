# Russell Stock Volume Prediction with Machine Learning

This project uses machine learning to predict Russell stock volume, spanning 7 steps from data analysis to model training, outlined in the Jupyter notebook below:

1.	Data Analysis
2.	Create predictors & Build dataset for models
3.	Dense model & LSTM model
4.	Analysis-dense & Analysis-lstm
5.	Adjust predictors & Build new dataset for models
6.	Enhancement-predictor & Enhancement-predictor
7.	Analysis-metrics & Analysis-feature importance
Other notebooks: 5. DQ & 7. Validation of Model Effectiveness

In notebooks 1, I addressed data quality issues such as duplicates and missing values, validated assumptions based on field definitions, and used the intersection of datasets to develop features/predictors. Notebooks 2 involved creating factors like technical signals, market capitalization, style and industry factors, and calendar/earnings release schedules.

In notebooks 3 and 4, I implemented Dense Neural Networks and LSTM Neural Networks based on the paper’s structure, training with cumulative and single-factor sets to evaluate feature selection based on model performance and feature importance. In notebooks 5&6&7, I adjusted predictors and model structures, analyzing results with the same method.

In 5. DQ, I checked the data issue with different ID types (BARRID and ISIN).
In 7. Validation of Model Effectiveness, I computed the MSE and R^2 using moving average values.

----------------------------

Other files:

1. writeup.pdf: the summary of the project.

2. html plots:
The feature importance plots:
(1). dense_feature_importance_single.html & dense_feature_importance_cumulative.html: Base Dense models
(2). rnn_feature_importance_single.html & rnn_feature_importance_cumulative.html: Base LSTM models
(3). final_feature_importance_plots.html: LSTM models with enhancement

3. python scripts:
These scripts are used in the Jupyter notebooks for model training and results analysis.

(1). model_training_utils.py: the helper functions for model training
(2). train_models.py: the functions to train models
(3). model_analysis.py: the functions to analyze model results

4. environment files:
(1). requirements.yaml: major packages
(2). environments.yaml: conda environment 

5. config files:
(1). config/final_dataset_column_names.json: the final dataset config
(2). config/final_dataset_column_namesv2.json: version 2 of the final dataset config

6. csv file:
factor_types.csv: the factor types in barra dataset

-----------------------------

### Steps to make new predictions out of sample:

#### Step 1. Build new predictors based on new data:
1. run codes in "2. Create predictors.ipynb" to generate initial files:
(1). run code before and in "[1] Find the universe after mapping" to generate final_universe
(2). run "2. Fundamental firm characteristics: market equity" to generate mkt_cap
(3). run "4. Calendar dates with large effects on trading volume (“calendar”)" to generate calendar_date
(4). run "5. Earnings release schedule (“earnings”):" to generate release_schedule

2. run all the codes in "5. Adjust predictors.ipynb" to generate final predictor values.

3. run all the codes in "5. Build new dataset for models.ipynb" to generate final predictor dataset called "pre_train_dataset" and "final dataset".

#### Step 2. Fit model and make prediction
run codes in "6. Enhancement-structure.ipynb"
(1). run code before and in "[1]. Preparation" to load dataset and prepare the train, validation and test dataset division.

(2). run codes in "2. Fit models to test..." and only keep test_num=5, which is the final model.

