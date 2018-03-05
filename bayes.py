from __future__ import division

from math import log
import operator
from sys import argv

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

USAGE = "python bayes.py TRAIN_FILE TEST_FILE MODE[n|t]"


##### NAIVE BAYES #####

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


##### TAN BAYES #####

## BASE FUNCTIONS

def tan_train(data, meta):
    # get the attr ranges dict
    attr_ranges = {}
    for feature_name, details in meta._attributes.iteritems():
        _, feature_range = details
        attr_ranges[feature_name] = feature_range
    # construct MST
    edge_weights = get_edge_weights(data, attr_ranges)
    edges = construct_tree(data, meta, attr_ranges, edge_weights)
    # generate probability dicts
    pd1, pd2 = tan_generate_prob_dicts(data, meta, edges, attr_ranges)
    class_probs = data['class'].value_counts()
    class_probs = class_probs.apply(lambda x: (x + 1) / (len(data) + 2))
    return pd1, pd2, class_probs, edges


def tan_generate_prob_dicts(data, meta, edges, attr_ranges):
    # we basically need two dicts
    # 1 --> p(x|c) for all x and c
    # 2 --> p(x|c,x.parent) for all x and c
    conditional_dict_1 = {}
    conditional_dict_2 = {}
    attribute_names = [x for x in data]
    labels = attr_ranges["class"]
    for attribute in attribute_names:
        attribute_values = attr_ranges[attribute]
        for label in labels:
            for value in attribute_values:
                # P (x/c)
                key = (attribute, value, label)
                conditional_dict_1[key] = get_conditional_prob(data, attr_ranges, attribute, value, label)
                # P (x|c,xp)
                parent = get_parent(attribute, edges)
                if parent:
                    for parent_value in attr_ranges[parent]:
                        key = (attribute, value, parent, parent_value, label)
                        conditional_dict_2[key] = get_conditional_prob_1x2(data, attr_ranges, attribute, value, parent,
                                                                           parent_value, label)

    return conditional_dict_1, conditional_dict_2


def tan_predict(data, meta, pd1, pd2, class_probs, edges):
    # print tree structure
    print_tree_structure(data, edges)
    print ""
    # begin prediction
    labels = meta._attributes['class'][1]
    attrs = [x for x in meta]
    count = 0
    for index, record in data.iterrows():
        record_label = record["class"]
        per_label_prob = {}
        for label in labels:
            this_label_prob = 1
            class_prob = class_probs[label]
            root_prob_key = (attrs[0], record[0], label)
            root_prob = pd1[root_prob_key]
            this_label_prob *= (class_prob * root_prob)
            for attr, attr_value in record.iteritems():
                parent = get_parent(attr, edges)
                if parent:
                    key = (attr, attr_value, parent, record[parent], label)
                    this_label_prob *= pd2[key]
            per_label_prob[label] = this_label_prob
        total_prob = sum(per_label_prob.values())
        final_probs = {k: v / total_prob for k, v in per_label_prob.iteritems()}
        predicted_label, certainty = max(final_probs.iteritems(), key=operator.itemgetter(1))
        print "%s %s %.12f" % (predicted_label, record_label, certainty)
        if predicted_label == record_label:
            count += 1
    print ""
    print count


## TREE CONSTRUCTION

def construct_tree(data, meta, attr_ranges, edge_weights):
    # uses Prim's algorithm to compute the tree
    # this is a fully connected graph.
    nodes = []
    edges = []
    attrs = [x for x in data]
    attrs.remove("class")
    # start by adding in the first attr into the nodelist
    nodes.append(attrs.pop(0))
    max_weight = -1
    new_dest_node = None
    new_source_node = None
    while attrs:
        for candidate_node in attrs:
            for existing_node in nodes:
                this_ew = edge_weights[(existing_node, candidate_node)]
                if this_ew > max_weight:
                    max_weight = this_ew
                    new_source_node = existing_node
                    new_dest_node = candidate_node
        # print "Linking %s -> %s" % (new_source_node, new_dest_node)
        nodes.append(new_dest_node)
        attrs.remove(new_dest_node)
        edges.append((new_source_node, new_dest_node))
        max_weight = -1
        new_dest_node = None
        new_source_node = None
    return edges


def get_single_edge_weight(data, attr_ranges, attribute1, attribute2):
    log2 = lambda x: log(x, 2)
    edge_weight = 0
    for a1_val in attr_ranges[attribute1]:
        for a2_val in attr_ranges[attribute2]:
            for y in attr_ranges["class"]:
                this_weight = 0
                # P(xi, xj, y)
                jp = get_joint_prob(data, attr_ranges, attribute1, a1_val, attribute2, a2_val, y)
                # P(xi, xj | y)
                cp = get_conditional_prob_2x1(data, attr_ranges, attribute1, a1_val, attribute2, a2_val, y)
                # P(xi | y)
                cp_xi = get_conditional_prob(data, attr_ranges, attribute1, a1_val, y)
                # P(xj | y)
                cp_xj = get_conditional_prob(data, attr_ranges, attribute2, a2_val, y)
                this_weight = jp * log2(cp / (cp_xi * cp_xj))
                edge_weight += this_weight
    return edge_weight


def get_edge_weights(data, attr_ranges):
    edge_weights = {}
    for attr1 in attr_ranges:
        for attr2 in attr_ranges:
            if attr1 == attr2:
                continue
            if "class" in [attr1, attr2]:
                continue
            edge_weights[(attr1, attr2)] = get_single_edge_weight(data, attr_ranges, attr1, attr2)
    return edge_weights


def print_tree_structure(data, edges):
    attrs = [x for x in data]
    classlabel = attrs.pop(-1)
    for attr in attrs:
        printstring = []
        printstring.append(attr)
        for src, dst in edges:
            if dst == attr:
                printstring.append(src)
        printstring.append(classlabel)
        print " ".join(printstring)


## HELPERS

def get_conditional_prob(data, attr_ranges, attribute1, a1_val, y):
    count = data[(data[attribute1] == a1_val) &
                 (data['class'] == y)].count()[0]
    total = data[data['class'] == y].count()[0]
    # laplace smoothing
    count += 1
    total += len(attr_ranges[attribute1])
    return count / total


def get_joint_prob(data, attr_ranges, attribute1, a1_val, attribute2, a2_val, y):
    count = data[(data[attribute1] == a1_val) &
                 (data[attribute2] == a2_val) &
                 (data['class'] == y)].count()[0]
    total = len(data)
    # laplace smoothing
    count += 1
    total += len(attr_ranges[attribute1]) * len(attr_ranges[attribute2]) * len(attr_ranges["class"])
    return count / total


def get_conditional_prob_1x2(data, attr_ranges, attr1, val1, attr2, val2, label):
    # returns P(attr1 | attr2,label)
    # Count(attr1, attr2, label)
    count = data[(data[attr1] == val1) &
                 (data[attr2] == val2) &
                 (data['class'] == label)].count()[0]
    # Count (attr1, label)
    total = data[(data[attr2] == val2) &
                 (data['class'] == label)].count()[0]
    # laplace smoothing
    count += 1
    total += len(attr_ranges[attr1])
    return count / total


def get_conditional_prob_2x1(data, attr_ranges, attribute1, a1_val, attribute2, a2_val, y):
    # Count(xi, xj, y)
    count = data[(data[attribute1] == a1_val) &
                 (data[attribute2] == a2_val) &
                 (data['class'] == y)].count()[0]
    # Count(y)
    total = data[data['class'] == y].count()[0]
    # laplace smoothing
    count += 1
    total += len(attr_ranges[attribute1]) * len(attr_ranges[attribute2])
    return count / total


def get_parent(attribute, edges):
    for src, dst in edges:
        if dst == attribute:
            return src
    return None


def print_debug(cd2, meta, attr_ranges):
    attrs = [x for x in meta]
    for k, v in cd2.iteritems():
        (attribute, value, parent, parent_value, label) = k
        attribute_number = attrs.index(attribute)
        parent_number = attrs.index(parent)
        value_number = attr_ranges[attribute].index(value)
        pvalue_number = attr_ranges[parent].index(parent_value)
        label_number = attr_ranges["class"].index(label)
        print "Pr(%d=%d | %d=%d,18=%d) = %.12f" % (attribute_number,
                                                   value_number,
                                                   parent_number,
                                                   pvalue_number,
                                                   label_number,
                                                   v)


if __name__ == "__main__":

    if len(argv) != 4:
        print USAGE

    train_file = argv[1]
    test_file = argv[2]
    mode = argv[3]

    train_data, train_meta = loadarff(file(train_file))
    train_data = pd.DataFrame(train_data)
    test_data, test_meta = loadarff(file(test_file))
    test_data = pd.DataFrame(test_data)

    if mode == "n":
        prob_dict, class_probs = nb_train(train_data, train_meta)
        nb_predict(test_data, test_meta, prob_dict, class_probs)

    elif mode == "t":
        pd1, pd2, class_probs, edges = tan_train(train_data, test_meta)
        tan_predict(test_data, test_meta, pd1, pd2, class_probs, edges)

    else:
        print USAGE
