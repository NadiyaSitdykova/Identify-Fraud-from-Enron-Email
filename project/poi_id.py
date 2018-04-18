#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

label = 'poi'
financial_features = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]
email_features = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]

features_list = [label] + financial_features + email_features 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Explore dataset
print("Total number of data points: " + str(len(data_dict.keys())))
print("Number of POIs: " + str(sum(data_dict[name]['poi'] for name in data_dict.keys())))

missing_values = dict(zip(features_list, [0 for _ in features_list]))
for feature in features_list:
    for name in data_dict.keys():
        if data_dict[name][feature] == 'NaN':
            missing_values[feature] += 1
        
print("Missing values by feature:")
for feature in features_list:
    print(feature + " " + str(missing_values[feature]))

### Explore outliers
def PlotScatter(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [feature_x, feature_y, label])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            col = 'red'
        else:
            col = 'green'
        plt.scatter(x, y, color=col)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

PlotScatter(data_dict, 'total_payments', 'total_stock_value')
PlotScatter(data_dict, 'salary', 'bonus')
PlotScatter(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')

### Remove meaningless records,outlier and data-entry error
records_to_remove = ['LOCKHART EUGENE E', 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for record in records_to_remove:
    data_dict.pop(record, 0)


### Create features
def compute_fraction(poi_messages, all_messages):
    """ return fraction of messages related to POI of all messages"""    
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 'NaN'
    fraction = poi_messages / all_messages
    return fraction

my_dataset = data_dict
for name in my_dataset.keys():
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    data_point['fraction_from_poi'] = fraction_from_poi
    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    data_point['fraction_to_poi'] = fraction_to_poi
    shared_receipt_with_poi = data_point['shared_receipt_with_poi']
    fraction_shared_with_poi = compute_fraction(shared_receipt_with_poi, to_messages)
    data_point['fraction_with_poi'] = fraction_shared_with_poi
    
features_list = features_list + ['fraction_from_poi', 'fraction_to_poi', 'fraction_with_poi']

### Get K-best features
from sklearn.feature_selection import SelectKBest
k = 5

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
d = dict(zip(features_list[1:], scores))
sorted_pairs = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
best_features = list(map(lambda x: x[0], sorted_pairs[:k]))

### Add other engineered features
extended_best_features = best_features + ['fraction from poi', 'fraction with poi']

### Extract features and labels from dataset for local testing
my_feature_list = [label] + extended_best_features
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Impute missing values with medians
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
features = imputer.fit_transform(features)

### Scale features via robust scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
features = scaler.fit_transform(features)

### Tuning: grid search
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'tol': [0.0001, 0.001, 0.1],'C': [0.01, 0.1, 1, 10, 100, 1000], 'class_weight':['balanced']} 
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(features_train, labels_train)
print(grid.best_params_)

### Creating classificator 
clf = LogisticRegression(tol=0.001, C=0.025, class_weight='balanced')

### My custom score
def classification_report_with_accuracy_score(y_true, y_pred):
    original.extend(y_true)
    predicted.extend(y_pred)
    return accuracy_score(y_true, y_pred) 

### Cross-validation of algorithm on 1000 different splits
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics.scorer import make_scorer
original = []
predicted = []
cv = StratifiedShuffleSplit(n_splits=1000, random_state=42)
score = cross_val_score(clf, features, labels, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score))

### Print average values in classification report for all folds in a K-fold Cross-validation  
print(classification_report(original, predicted)) 

### Dumping my classifier, dataset, and features_list 

dump_classifier_and_data(clf, my_dataset, features_list)
