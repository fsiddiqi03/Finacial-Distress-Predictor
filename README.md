# Finacial Distress Project Report
### Project Goal
The goal of this project is to predict the likelihood of borrowers experiencing financial distress within the next two years, leading to serious defaults or bankruptcy. The prediction is based on borrower features such as age, monthly income, debt, number of times late on payment, and past payment history. The target variable, “SeriousDlqin2yrs,” is binary, with 1 indicating serious financial distress and 0 indicating no such distress.

### Task and Experience
The primary task is to predict whether borrowers will experience financial distress within the next two years. This project involves data preprocessing, model training, and evaluation using various machine learning algorithms like logistic regression, K-Nearest Neighbors (KNN), and decision trees. The focus is on using borrower features to predict defaults accurately.

## Performance Metrics
To evaluate model performance, several metrics are used:
- Recall: Recall, also known as sensitivity or true positive rate, measures the proportion of actual positive instances that were correctly predicted by the model. 
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
- With the data preprocessing and splitting complete, the next step involves training the models. 

![image](https://github.com/fsiddiqi03/Finacial-Distress-Predictor/assets/126859213/95bbe854-78a7-46c5-a030-8e5972bf5e94)

```bash
Cumulative Explained Variance: [0.30547614 0.46780693 0.58797384 0.6887143  0.78877704 0.87830108
 0.94906308 0.99886648 0.99964287 1.]
```

## Training the Models 
In the model training section, I trained four different models: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest.

### Logistic Regression 

#### Model Training 
- X_train_scaled represents the standardized features of the balanced training dataset, where each feature has been scaled to have a mean of 0 and a standard deviation of 1.
- y_train_resampled denotes the target variable of the training data after oversampling the minority class using SMOTE (Synthetic Minority Over-sampling Technique), ensuring a more balanced distribution of classes.
- The Logistic Regression model was initialized with class_weight='balanced' to adjust for any remaining class imbalance, assigning higher weights to minority class samples during model training.
- Additionally, the model was trained using a specified random_state=42 for reproducibility, ensuring consistent results across different runs.

```python
# Create a Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)  # 'balanced' to address any remaining imbalance

# Train the model on the balanced training data
model.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)
```

#### Model Metrics
```bash
Precision: 0.9067818916275341
Recall: 0.6704
F1-score: 0.7538060324496375
AUC score: 0.6455974529070444
```
- Precision: The precision is approximately 0.907, indicating that around 90.7% of the positive predictions made by the model are correct.
- Recall: The recall is approximately 0.6704, indicating that the model correctly identifies around 67.04% of all actual positive instances.
- F1-score: The  F1-score is approximately 0.754, indicating a good balance between precision and recall.
- AUC score: The AUC score is approximately 0.646, suggesting moderate discriminative ability of the model.


### KNN

#### Overview
K-Nearest Neighbors (KNN) is a non-parametric and instance-based learning algorithm used for classification and regression tasks. It operates on the principle of similarity, where the class label of a new instance is determined by the majority class among its k nearest neighbors in the feature space.

#### Model Training 
- X_train_scaled: Standardized features of the training dataset, ensuring that each feature has a mean of 0 and a standard deviation of 1, making them comparable.
- y_train_resampled: Target variable of the training data after employing Synthetic Minority Over-sampling Technique (SMOTE), which balances the class distribution by generating synthetic samples of the minority class.
- KNN Classifier Configuration: The KNN model is configured with parameters such as the number of neighbors (n_neighbors), distance metric (metric), and algorithm for nearest neighbor search (algorithm).
  - n_neighbors: The number of neighbors considered when making predictions.
  - metric: The distance metric used to measure the similarity between instances. Common choices include Euclidean distance (‘euclidean’), Manhattan distance (‘manhattan’), or Minkowski distance (‘minkowski’).
  - algorithm: The algorithm used to compute the nearest neighbors. Options include brute-force search (‘brute’), ball tree (‘ball_tree’), or KD tree (‘kd_tree’).

```python
# Algorithims
brute = 'brute'
ball = 'ball_tee'
kd = 'kd_tree'

# Metrics
man = 'manhattan'
min = 'minkowski'

# KNN classifier configuration
knn = KNeighborsClassifier(n_neighbors=15, metric=min, p=2, algorithm=brute)

# Train the KNN classifier
knn.fit(X_train_scaled, y_train_resampled)

# Predict with KNN on the test data
predictions_knn = knn.predict(X_test_scaled)
```

#### Confusion Matrix
The confusion matrix is a table that summarizes the performance of a classification model by comparing actual and predicted class labels. It provides insights into the model's ability to correctly classify instances into true positive (TP), true negative (TN), false positive (FP), and false negative (FN) categories.

![image](https://github.com/fsiddiqi03/Finacial-Distress-Predictor/assets/126859213/e8bfa91e-20b9-488d-a82c-4261b630ca1a)

- 0,0 represents true positives (TP), where the actual class and the predicted class are both positive. It's the value 23890.
- 1,1 represents true negatives (TN), where the actual class and the predicted class are both negative. It's the value 885.
- 0,1 represents false positives (FP), where the actual class is negative but predicted as positive. It's the value 4154.
- 1,0 represents false negatives (FN), where the actual class is positive but predicted as negative. It's the value 1071.

#### Metrics
```bash
Precision: 0.90614167889564
Recall: 0.8258333333333333
F1-score: 0.8591495934912204
AUC score: 0.6521648058747297
```
- Precision: Approximately 90.61% of the positive predictions made by the model are correct. This indicates a high proportion of correctly predicted positive instances out of all instances predicted as positive.
- Recall: Around 82.58% of all actual positive instances are correctly identified by the model. This suggests that the model captures a significant portion of positive instances out of all actual positive instances.
- F1-score: The F1-score, which combines precision and recall, is approximately 85.91%. It indicates a good balance between precision and recall, considering both false positives and false negatives.
- AUC score: The AUC score is approximately 0.652, indicating moderate discriminative ability of the model. It represents the area under the receiver operating characteristic (ROC) curve and measures the model's ability to distinguish between positive and negative classes.

### Random Forrest
Random Forest is an ensemble learning technique that combines multiple decision trees to create a more robust and accurate predictive model. Each tree in the forest is trained on a random subset of the data and makes independent predictions. The final prediction is then determined by averaging or voting across all trees. This approach helps reduce overfitting and improves generalization performance.

#### Model Training 
- The Random Forest classifier was trained using a randomized search for hyperparameter tuning, exploring various combinations to maximize the area under the ROC curve (AUC).
- Hyperparameters such as the number of estimators (trees), maximum depth of each tree, and the maximum number of features considered for splitting were optimized to enhance the model's predictive performance.
- The param_grid specifies the hyperparameters that will be explored during the randomized search for the Random Forest classifier. Here's an explanation of each parameter:
  - n_estimators: This parameter determines the number of trees in the forest. In the grid, we're considering 50, 100, and 150 trees.
  - max_depth: It defines the maximum depth of each tree in the forest. The value None means that nodes are expanded until all leaves are pure or until they contain less than the minimum samples required for a split.
  - max_features: This parameter specifies the maximum number of features considered for splitting a node. In the grid, we're exploring 1, 3, and 10 features.
  - min_samples_split: It determines the minimum number of samples required to split an internal node. The grid contains values 2, 3, and 10, representing the minimum number of samples required to split an internal node.

```python
# Use a full grid over all parameters
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, None],
    "max_features": [1, 3, 10],
    "min_samples_split": [2, 3, 10]
}

# Run grid search
grid = RandomizedSearchCV(clf, param_grid, cv=5, scoring="roc_auc", n_iter=10, random_state=42)
grid.fit(X_train_scaled, y_train_resampled)
```
#### Metrics 
- Precision: Around 91.2% of the positive predictions made by the model are correct, indicating high precision.
- Recall: The model correctly identifies around 89% of all actual positive instances, demonstrating good sensitivity.
- F1-score: With an F1-score of approximately 0.900, the model achieves a balance between precision and recall, signifying robust performance.
- AUC Score: The AUC score provides a measure of the model's ability to discriminate between positive and negative instances, with a value of approximately 0.805 indicating moderate discriminative ability.











