#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)

# no of poi's in test set
print labels_test.count(1)

# no of people in test set
print len(labels_test)

# counting true positives
def count_positives_negatives(true_labels, predictions, val):
    count = 0;
    for i in range(0,len(true_labels)):
        if true_labels[i] == val and true_labels[i] == predictions[i]:
            count += 1
    return count

print "true postitives ", count_positives_negatives(labels_test, pred, 1)

# precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print "true postitives ", count_positives_negatives(true_labels, predictions, 1)

# true negatives

print "true negatives ", count_positives_negatives(true_labels, predictions, 0)

print "precision ", precision_score(true_labels, predictions)

print "recall ", recall_score(true_labels, predictions)
