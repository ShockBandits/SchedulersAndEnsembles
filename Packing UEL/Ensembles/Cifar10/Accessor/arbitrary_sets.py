from collections import defaultdict
import cPickle
import numpy as np
import operator
import os
from random import shuffle
import sys

default_path = '/media/innovationcommons/DataStorage/Cifar-10/cifar-10-batches-py/'


def get_chosen_classes(chosen_list, source_batches):
    """ Takes Cifar10 data batches, and removes all classes except
        those on chosen_list. Returns a dict, whose first key is the
        source batch and secondary keys point to the Cifar10 data"""
    
    chosen_dict = defaultdict(dict)
    for curr_batch in source_batches:
        with open(os.path.join(default_path, curr_batch), 'rb') as f:
            cifar_dict = cPickle.load(f)

            data_list = [ctr
                         for curr_class in chosen_list
                         for ctr, x in enumerate(cifar_dict['labels'])
                         if x == curr_class]
            c_data = np.zeros((len(data_list), 3072),
                              dtype=cifar_dict['data'].dtype)
            c_labels = [0] * len(data_list)
            c_filenames = [''] * len(data_list)

            for ctr, row in enumerate(data_list):
                c_data[ctr, :] = cifar_dict['data'][row, :]
                c_labels[ctr] = cifar_dict['labels'][row]
                c_filenames[ctr] = cifar_dict['filenames'][row]

        chosen_dict[curr_batch]['data'] = c_data
        chosen_dict[curr_batch]['labels'] = c_labels
        chosen_dict[curr_batch]['filenames'] = c_filenames
    chosen_dict['meta_data'] = [chosen_list]
    return chosen_dict


def combine_batches(chosen_dict):
    """ Combines data from get_chosen_classes by removing primary keys
    (i.e. batch identifiers) from chosen dict and concatenating data
    in secondary keys"""

    batches = set(sorted(chosen_dict.keys())) - {'meta_data'}
    batches = sorted(list(batches))
    root_dict = dict()
    root_dict['data'] = chosen_dict[batches[0]]['data']
    root_dict['labels'] = chosen_dict[batches[0]]['labels']
    root_dict['filenames'] = chosen_dict[batches[0]]['filenames']
    root_dict['meta_data'] = chosen_dict['meta_data']
    root_dict['meta_data'].append(batches[0])

    for curr_batch in batches[1:]:
        temp_dict = chosen_dict[curr_batch]
        root_dict['data'] = np.concatenate((root_dict['data'],
                                            temp_dict['data']),
                                           axis=0)
        root_dict['labels'] = root_dict['labels'] + temp_dict['labels']
        root_dict['filenames'] = root_dict['filenames'] + temp_dict['filenames']
        root_dict['meta_data'].append(curr_batch)

    tot_rows = root_dict['data'].shape[0]
    new_order = range(tot_rows)
    for _ in range(5):
        shuffle(new_order)

    ub_dict = dict()
    ub_data = np.zeros((tot_rows, 3072), dtype=root_dict['data'].dtype)
    ub_labels = [0] * tot_rows
    ub_filenames = [""] * tot_rows

    for ctr, idx in enumerate(new_order):
        ub_data[ctr, :] = root_dict['data'][idx, :]
        ub_labels[ctr] = root_dict['labels'][idx]
        ub_filenames[ctr] = root_dict['filenames'][idx]

    ub_dict['data'] = ub_data
    ub_dict['labels'] = ub_labels
    ub_dict['filenames'] = ub_filenames
    ub_dict['meta_data'] = root_dict['meta_data']

    return ub_dict


def set_class_distribution(ub_dict, percentage_dict, name):
    """ Takes data in ub_dict and rebalances it so that the class distribution
        matches that given by percentage_dict. Each key,value pair in percentage
        dict is class number, percentage of distribution. The name variable
        gives the name of the file to which the data will be saved"""
    tot_percent = 0
    for x in percentage_dict:
        tot_percent += percentage_dict[x]
    label_ctr_dict = defaultdict(int)
    for x in ub_dict['labels']:
        label_ctr_dict[x] += 1
        
    if tot_percent != 1:
        sys.exit("Total percentages != 1")
    if len(ub_dict['meta_data'][0]) != len(percentage_dict):
        sys.exit("Mismatch between expected and given number of classes")
    if set(ub_dict['meta_data'][0]) != set(percentage_dict):
        sys.exit("Mismatch between classes given and those expected")

    batch_size = int(min([label_ctr_dict[x]/percentage_dict[x] for x in percentage_dict]))
    class_trgt_distrib = {x: int(batch_size*percentage_dict[x]) for x in percentage_dict}
    class_actual_distrib = {x: 0 for x in percentage_dict}
    tot_trgts = sum([class_trgt_distrib[x] for x in class_trgt_distrib])
    if  tot_trgts < batch_size:
        key, val = min(class_trgt_distrib.iteritems(), key=operator.itemgetter(1))
        class_trgt_distrib[key] += (batch_size - tot_trgts)

    tot_rows = batch_size

    bal_dict = dict()
    bal_data = np.zeros((tot_rows, 3072), dtype=ub_dict['data'].dtype)
    bal_labels = [0] * tot_rows
    bal_filenames = [""] * tot_rows

    bal_ctr = 0
    for idx in range(len(ub_dict['labels'])):
        curr_label = ub_dict['labels'][idx]
        if class_actual_distrib[curr_label] < class_trgt_distrib[curr_label]:
            bal_data[bal_ctr, :] = ub_dict['data'][idx, :]
            bal_labels[bal_ctr] = ub_dict['labels'][idx]
            bal_filenames[bal_ctr] = ub_dict['filenames'][idx]
            
            bal_ctr += 1
            class_actual_distrib[curr_label] += 1

    bal_dict['data'] = bal_data
    bal_dict['labels'] = bal_labels
    bal_dict['filenames'] = bal_filenames
    bal_dict['name'] = name
    bal_dict['src_meta_data'] = ub_dict['meta_data']

    return bal_dict


def shuffle_data_set(in_dict):
    """ Randomizes image order in data set"""
    tot_rows = in_dict['data'].shape[0]
    new_order = range(tot_rows)
    for _ in range(5):
        shuffle(new_order)

    out_dict = in_dict
    out_data = np.zeros((tot_rows, 3072), dtype=in_dict['data'].dtype)
    out_labels = [0] * tot_rows
    out_filenames = [""] * tot_rows

    for ctr, idx in enumerate(new_order):
        out_data[ctr, :] = in_dict['data'][idx, :]
        out_labels[ctr] = in_dict['labels'][idx]
        out_filenames[ctr] = in_dict['filenames'][idx]

    out_dict['data'] = out_data
    out_dict['labels'] = out_labels
    out_dict['filenames'] = out_filenames

    return out_dict


'''
default_path = '/media/innovationcommons/DataStorage/Cifar-10/cifar-10-batches-py/'
chosen_classes = [0, 4, 7]
y = get_chosen_classes(chosen_classes, ['data_batch_1', 'data_batch_2'])
yy = combine_batches(y)
perc_dict = {0: 0.25, 4: 0.50, 7: 0.25}
y3 = set_class_distribution(yy, perc_dict, 'test')
y4 = shuffle_data_set(y3)
'''