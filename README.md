
# PREDICTIVE ANALYSIS OF EARLY READMISSION

To predict the readmission of hospital patients in 30 days. Readmission of patients following discharge from hospital places an enormous burden on the US healthcare system. The objective of predictive analysis is to build a binary classification model that can predict early readmission given the patient’s features.




## Data
## Imports

```http
import pandas
import matplotlib.pyplot
import numpy
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve 
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

```




## Feature Engineering
- Dropped features that are not related to readmission of patients.
- Created new column ‘patient_visit_beforeadmit‘ by adding  three columns 'number_outpatient’, 'number_emergency','number_inpatient’  to store the number of times patient visited hospital in previous year.
- From 24 medication features we saw that some have all the values as ‘No’ and some have majority (99.9%) values as ‘No’ so we drop those features. From the remaining features, created a new column ‘num_medschange’ to store the total number of medications changed for a patient.
- ‘diag_1’, ‘diag_2’, ‘diag_3’ columns have many codes ranging from 1-954. So, we group together codes which mean the same and assigning a category to them.
- Some features are grouped together which have the same meaning and value.
- Conducted ONE-HOT ENCODING and STANDARDIZATION to create a final data set.




## Challenge - SMOTE

Our data consisted majority of data points wherein patients did not readmit in 30 days. There is a problem of class imbalance in the test and train dataset. We used SMOTE(Synthetic Minority Oversampling Technique) to counter this which is an effective way of synthesizing new samples from existing ones. 

```http
Original dataset shape: ({0: 86986, 1: 11066})
New dataset shape: ({0: 86986, 1: 86986})
```

## Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest


## Optimizations

Refinement Over Random Forest
- Cross Validation: number of splits =5
- Hyperparameter Tuning using RandomizedCV: 
```http
Optimize these 3 parameters and find best value from the list.
    space['n_estimators'] = [10, 100, 500]
    space['max_depth'] = [20, 25, 30]
    space['min_samples_split']=[8,10,12]
Random search across 25 candidates, using 5-fold cross validation.
    Parameters passed : n_iter = 25, cv = 5
```






## Results

- Logistic Regression has best TPR value, but there is a tradeoff between model complexity and accuracy. If we can compromise on FPR and TNR then Logistic Regression is good but if we choose Random Forest, we get better performance, Accuracy, although it is computationally costly.

- For our problem statement, Recall score is an effective measure. It is more preferable to not miss any patient who will be readmitted in 30 days even if that means predicting some patients as readmitted within 30days but actually they will not be readmit in 30 days.
