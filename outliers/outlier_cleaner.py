#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    all_data = []
    for i in range(0,len(ages)):
        single_row = []
        single_row.append(ages[i])
        single_row.append(net_worths[i])
        single_row.append(abs(predictions[i]-net_worths[i]))
        all_data.append(tuple(single_row))
    all_data.sort(key=lambda tup: tup[2])
    cleaned_data = all_data[:81]
    return cleaned_data

