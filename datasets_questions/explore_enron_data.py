#!/usr/bin/python

"""
    starter code for exploring the Enron dataset (emails + finances)
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)
print len(enron_data["SKILLING JEFFREY K"])

print enron_data["LAY KENNETH L"]

count = 0
count_poi = 0
for x in enron_data:
  if enron_data[x]['poi']:
    count_poi += 1
    if enron_data[x]['total_payments'] == 'NaN':
        count += 1
print count

print float(count)/count_poi
