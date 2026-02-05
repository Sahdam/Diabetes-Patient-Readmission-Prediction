# Diabetes Patient Readmission Prediction

## Project Overview
This project aims to predict the readmission of diabetes patients using a dataset containing various patient and encounter information. The goal is to develop a machine learning model, specifically a Multi-Layer Perceptron (MLP) using PyTorch, to classify whether a patient will be readmitted within 30 days (`<30`), after 30 days (`>30`), or not at all (`NO`).

## Dataset
The dataset used is `diabetic_data.csv`, which contains 50 features and 101,766 entries. It includes demographic information, hospital visit details, medical specialties, lab procedures, medications, diagnoses, and readmission status.

### Key Features:
*   `encounter_id`, `patient_nbr`
*   `race`, `gender`, `age`
*   `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
*   `time_in_hospital`
*   `payer_code`, `medical_specialty`
*   `num_lab_procedures`, `num_procedures`, `num_medications`
*   `number_outpatient`, `number_emergency`, `number_inpatient`
*   `diag_1`, `diag_2`, `diag_3`, `number_diagnoses`
*   `max_glu_serum`, `A1Cresult`
*   Various medication features (e.g., `metformin`, `insulin`, `glipizide`)
*   `change`, `diabetesMed`, `readmitted` (target variable)

## Data Preprocessing
The following steps were performed to prepare the data for modeling:

1.  **Age Group Mapping**: The `age` column, originally in range format (e.g., `[0-10)`), was mapped to a more interpretable string format (e.g., `0-9`).
2.  **Handling Missing Values**: 
    *   `'?'` values across the dataset were replaced with `np.nan`.
    *   Missing values in `max_glu_serum` and `A1Cresult` were filled with `'None'`.
    *   `'Unknown'` and `'Unknown/Invalid'` values in the `gender` column were replaced with `np.nan`.
    *   Rows with any remaining `np.nan` values were dropped from the selected features.
3.  **Feature Selection**: A subset of features was selected for the model:
    `['age', 'gender', 'time_in_hospital', 'number_inpatient', 'number_emergency', 'number_outpatient', 'num_medications', 'num_lab_procedures', 'number_diagnoses', 'A1Cresult', 'max_glu_serum', 'insulin', 'diabetesMed', 'change', 'readmitted']`
4.  **Categorical Feature Encoding**: Categorical features (`age`, `gender`, `A1Cresult`, `max_glu_serum`, `insulin`, `diabetesMed`, `change`) were one-hot encoded using `OneHotEncoder`.
5.  **Numerical Feature Scaling**: Numerical features were scaled using `StandardScaler`.
6.  **Target Variable Encoding**: The `readmitted` target variable was label encoded into numerical classes (0, 1, 2).

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of variables and their relationship with the target variable, `readmitted`.

*   **Distribution of `readmitted`**: Analyzed the counts of `NO`, `>30`, and `<30` readmissions.
*   **Categorical Feature Analysis**: Count plots were generated to visualize the distribution of `insulin`, `diabetesMed`, `age`, `gender`, `max_glu_serum`, and `change` against `readmitted`.
*   **Numerical Feature Analysis**: Bar plots were created to show the frequency of readmissions (`<30`) across different values for `time_in_hospital`, `number_emergency`, `number_inpatient`, `number_outpatient`, `number_diagnoses`, `num_medications`, and `num_lab_procedures`.
*   **Correlation Analysis**: A heatmap was used to visualize the correlation between numerical features.

## Model Training

The data was split into training (80%), validation (10%), and test (10%) sets using `train_test_split` with stratification to maintain class distribution. Several MLP models were implemented and trained using PyTorch:

1.  **Base MLP**: A simple MLP with one hidden layer and ReLU activation.
2.  **MLP with Batch Normalization (DiabetesBN_MLP)**: Added `BatchNorm1d` layer after the first linear layer to improve training stability and performance.
3.  **MLP with Dropout and L2 Regularization (Diabetes_MLP_Drop_L2)**: Incorporated `Dropout` for regularization and `weight_decay` in the optimizer for L2 regularization.
4.  **MLP with Class Weights (Diabetes_MLP_CW)**: Used `CrossEntropyLoss` with class weights calculated from the training data to address class imbalance.
5.  **Multi-layer MLP with BCEWithLogitsLoss (DiabetesMLP)**: A deeper MLP with two hidden layers and `BCEWithLogitsLoss` using `pos_weight` for binary classification. *(Note: This model was trained for a binary classification task, which might not align with the 3-class target variable. The `pos_weight` calculation in the notebook assumes binary classification.)*

All models were trained for 1000 epochs, and training/validation loss was monitored.

## Model Evaluation

Models were evaluated on the test set using Accuracy and Macro F1-score. For the best-performing model (Diabetes_MLP_CW), a detailed classification report and a confusion matrix were generated.

### Evaluation Metrics:
*   **Accuracy Score**: The proportion of correctly classified instances.
*   **Macro F1 Score**: The unweighted mean of the F1 score for each class, useful for imbalanced datasets.

### Model Performance Comparison:
| Model    | Test Accuracy | Macro F1 Score |
| :------- | :------------ | :------------- |
| Base     | 0.568832      | 0.379697       |
| Drop_out | 0.569913      | 0.361146       |
| CW       | 0.480692      | 0.416107       |
| CW_new   | 0.345485      | 0.261711       |

### Classification Report for Diabetes_MLP_CW:
```
              precision    recall  f1-score   support

           0       0.19      0.43      0.26      1136
           1       0.43      0.30      0.36      3555
           2       0.66      0.61      0.63      5486

    accuracy                           0.48     10177
   macro avg       0.43      0.45      0.42     10177
weighted avg       0.53      0.48      0.49     10177
```

## Results and Discussion

The `Diabetes_MLP_CW` model, which uses class weighting to handle imbalance, showed the best Macro F1-score despite having a lower overall accuracy compared to the `Base` and `Drop_out` models. This suggests that while overall accuracy might be lower, it is doing a better job at identifying the minority classes, which is often crucial in medical prediction tasks. The individual class metrics in the classification report highlight the challenge in predicting the minority classes (0 and 1, representing readmission `<30` and `>30` respectively) compared to the majority class (2, representing `NO` readmission).

Further efforts could focus on advanced techniques for imbalanced datasets, more complex model architectures, or hyperparameter tuning to improve overall performance, especially for the `<30` readmission class.
```
