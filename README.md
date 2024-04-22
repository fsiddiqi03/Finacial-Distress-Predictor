# Finacial Distress Project Report
### Project Goal
The goal of this project is to predict the likelihood of borrowers experiencing financial distress within the next two years, leading to serious defaults or bankruptcy. The prediction is based on borrower features such as age, monthly income, debt, number of times late on payment, and past payment history. The target variable, “SeriousDlqin2yrs,” is binary, with 1 indicating serious financial distress and 0 indicating no such distress.

### Task and Experience
The primary task is to predict whether borrowers will experience financial distress within the next two years. This project involves data preprocessing, model training, and evaluation using various machine learning algorithms like logistic regression, K-Nearest Neighbors (KNN), and decision trees. The focus is on using borrower features to predict defaults accurately.

## Performance Metrics
To evaluate model performance, several metrics are used:
- Accuracy: Overall correctness of the model.
- Precision and Recall: Important due to class imbalance in the dataset, with precision indicating accuracy of positive predictions, and recall indicating the proportion of actual positives correctly identified.
- F1 Score: The harmonic mean of precision and recall, providing a single score to assess model performance.
- ROC-AUC: Measures the model's ability to distinguish between classes across all thresholds.

## Data Preprocessing and Splitting
- Here are the steps taken to preprocess and split the data:
- Loading and Checking the Data:
  -  The data was loaded from a CSV file
  -  After loading, the first few rows were examined to understand the dataset structure.

### Handling Missing Values:
- Two columns had missing values: "MonthlyIncome" and "NumberOfDependents".
- "MonthlyIncome" was imputed with the median value, while "NumberOfDependents" was imputed with the mode.
  
### Data Splitting:
- The data was split into training and test sets using an 80-20 split.
- Features were separated from the target variable ("SeriousDlqin2yrs").
  
### Dealing with Class Imbalance:
- Given the class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the training data.
- This increased the proportion of positive samples in the training set from about 6.7% to 50%.

### Standardization:
- The training and test data were standardized using StandardScaler to ensure consistency among features.

### PCA Check:
- Principal Component Analysis (PCA) was applied to check the cumulative explained variance.
- The results showed a good distribution of variance across features, leading to the decision not to use PCA in further analysis.
- With the data preprocessing and splitting complete, the next step involves training the logistic regression model. This section will detail the code and methodology used for this process.

