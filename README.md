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
 

### Data Splitting:
- The data was split into training and test sets using an 80-20 split.
- Features were separated from the target variable ("SeriousDlqin2yrs").

```python
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

# Load data
data_df = pd.read_csv('cs-training.csv')

# Drop 'Unnamed: 0' if necessary
data_df.drop(columns='Unnamed: 0', inplace=True)

# Separate features and target
X = data_df.drop(columns='SeriousDlqin2yrs')
y = data_df['SeriousDlqin2yrs']

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train.value_counts()[1]/X_train.shape[0])
print(y_test.value_counts()[1]/X_test.shape[0])


# train info
print('Train data shape:', X_train.shape)
print('Train data labels, label count:', np.unique(y_train, return_counts=True))
print()

# test info
print('Test data shape:', X_test.shape)
print('Test data labels, label count:', np.unique(y_test, return_counts=True))
print()
```


### Handling Missing Values:
- Two columns had missing values: "MonthlyIncome" and "NumberOfDependents".
- "MonthlyIncome" was imputed with the median value, while "NumberOfDependents" was imputed with the mode.

 ```python
# Impute 'MonthlyIncome' with the median
income_imputer = SimpleImputer(strategy='median')
X_train['MonthlyIncome'] = income_imputer.fit_transform(X_train[['MonthlyIncome']])

# Impute 'NumberOfDependents' with the mode (most frequent value)
dependents_imputer = SimpleImputer(strategy='most_frequent')
X_train['NumberOfDependents'] = dependents_imputer.fit_transform(X_train[['NumberOfDependents']])

# Apply the same imputation to the test set
X_test['MonthlyIncome'] = income_imputer.transform(X_test[['MonthlyIncome']])
X_test['NumberOfDependents'] = dependents_imputer.transform(X_test[['NumberOfDependents']])

# Check for missing values in the training data
print("\nMissing Values in Training Data:")
print(X_train.isnull().sum())

# Check for missing values in the test data
print("\nMissing Values in Training Data:")
print(X_test.isnull().sum())
   ```
  
### Dealing with Class Imbalance:
- Given the class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the training data.
- This increased the proportion of positive samples in the training set from about 6.7% to 50%.

```python
# before SMOTE
print(y_train.value_counts()[1] / X_train.shape[0])

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Verify the new class distribution
print('Train data shape after SMOTE:', X_train_resampled.shape)
print('Train data labels after SMOTE, label count:', np.unique(y_train_resampled, return_counts=True))

# after SMOTE
print(y_train_resampled.value_counts()[1]/X_train_resampled.shape[0])
```


### Standardization:
- The training and test data were standardized using StandardScaler to ensure consistency among features.

```python
# Standardize the resampled training data
scaler = StandardScaler()
X_train_resampled_standardized = scaler.fit_transform(X_train_resampled)
X_test_resampled_standardized = scaler.transform(X_test)
```

### PCA Check:
- Principal Component Analysis (PCA) was applied to check the cumulative explained variance.
- The results showed a good distribution of variance across features, leading to the decision not to use PCA in further analysis.
- With the data preprocessing and splitting complete, the next step involves training the logistic regression model. This section will detail the code and methodology used for this process.

