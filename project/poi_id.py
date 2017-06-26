#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data

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


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
"""from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import recall_score, precision_score
print("recall is " + str(recall_score(labels_test, pred)))
print("precision is " + str(precision_score(labels_test, pred)))
print(pred)"""

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
