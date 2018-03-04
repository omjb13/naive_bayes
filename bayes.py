from __future__ import division

import operator
from sys import argv

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

USAGE = "python bayes.py TRAIN_FILE TEST_FILE MODE[n|t]"


def nb_train(data, meta):
    # build out the probability dictionary
    # key => "(attribute_name, attribute_value, label)"
    # value => P(attr_value|label) (with laplace smoothing)
    prob_dict = {}
    # get possible attribute values from meta instead of
    # looking at data directly as there may be
    # feature values that are missing
    attr_ranges = {}
    for feature_name, details in meta._attributes.iteritems():
        _, feature_range = details
        attr_ranges[feature_name] = feature_range
    attribute_names = [x for x in data]
    # get probs for classes
    class_probs = data['class'].value_counts()
    class_probs = class_probs.apply(lambda x: (x + 1) / (len(data) + 2))
    # begin training
    labels = data['class'].unique()
    for attribute in attribute_names:
        attribute_values = attr_ranges[attribute]
        for label in labels:
            for value in attribute_values:
                if attribute == 'class':
                    continue
                key = (attribute, value, label)
                count = data.loc[(data[attribute] == value) & (data['class'] == label)].count()[0]
                class_count = data[data['class'] == label].count()[0]
                # laplace smoothing
                count += 1
                class_count = class_count + len(attribute_values)
                prob = np.float64(count / class_count)
                prob_dict[key] = prob
    return prob_dict, class_probs


def nb_predict(data, meta, prob_dict, class_probs):
    # Print structure
    metalist = list(meta)
    for name in metalist[:-1]:
        print name, metalist[-1]
    print ""
    # Prediction
    labels = meta._attributes['class'][1]
    count = 0
    for index, record in data.iterrows():
        record_label = record['class']
        per_label_prob = {}
        for label in labels:
            class_prob = class_probs[label]
            this_label_prob = class_prob
            for attr, attr_value in record.iteritems():
                if attr == 'class':
                    continue
                key = (attr, attr_value, label)
                this_label_prob *= prob_dict[key]
            per_label_prob[label] = this_label_prob
        total_prob = sum(per_label_prob.values())
        final_probs = {k: v / total_prob for k, v in per_label_prob.iteritems()}
        predicted_label, certainty = max(final_probs.iteritems(), key=operator.itemgetter(1))
        print "%s %s %.12f" % (predicted_label, record_label, certainty)
        if predicted_label == record_label:
            count += 1
    print ""
    print count


if __name__ == "__main__":

    if len(argv) != 4:
        print USAGE

    train_file = argv[1]
    test_file = argv[2]
    mode = argv[3]
    mode = "n"

    train_data, meta = loadarff(file(train_file))
    train_data = pd.DataFrame(train_data)
    test_data, meta = loadarff(file(test_file))
    test_data = pd.DataFrame(test_data)

    if mode == "n":
        # train NB
        prob_dict, class_probs = nb_train(train_data, meta)

        # predict and print
        nb_predict(test_data, meta, prob_dict, class_probs)

    elif mode == "t":
        pass

    else:
        print USAGE
