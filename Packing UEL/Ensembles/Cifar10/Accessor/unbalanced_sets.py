from collections import defaultdict
import cPickle
import numpy as np
import os
from random import random, shuffle

default_path = '/media/innovationcommons/DataStorage/Cifar-10/cifar-10-batches-py/'


def get_overbalanced_classes(over_list, source_batches):

    over_dict = defaultdict(dict)
    for curr_batch in source_batches:
        with open(os.path.join(default_path, curr_batch), 'rb') as f:
            cifar_dict = cPickle.load(f)
            data_list = [ctr
                         for curr_class in over_list
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

        over_dict[curr_batch]['data'] = c_data
        over_dict[curr_batch]['labels'] = c_labels
        over_dict[curr_batch]['filenames'] = c_filenames
    return over_dict


def make_unbalanced_dict(root_batch, over_dict, ub_name):

    with open(os.path.join(default_path, root_batch), 'rb') as f:
        root_dict = cPickle.load(f)

    for curr_batch in over_dict:
        temp_dict = over_dict[curr_batch]
        root_dict['data'] = np.concatenate((root_dict['data'],
                                            temp_dict['data']),
                                           axis=0)
        root_dict['labels'] = root_dict['labels'] + temp_dict['labels']
        root_dict['filenames'] = root_dict['filenames'] + temp_dict['filenames']

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
    ub_dict['batch_label'] = ub_name

    return ub_dict


def filter_classes(in_dict, safe_classes, filter_prob):
    filtered_rows = [ctr for ctr, x in enumerate(in_dict['labels'])
                     if ((x not in safe_classes) and
                         (random() < filter_prob))]
    tot_rows = len(in_dict['labels'])
    num_filtered = len(filtered_rows)
    fd_rows = tot_rows - num_filtered
    filtered_dict = dict()
    filtered_data = np.zeros((fd_rows, 3072),
                             dtype=in_dict['data'].dtype)
    filtered_labels = [0] * fd_rows
    filtered_filenames = [""] * fd_rows

    filt_ctr = 0
    for ctr in range(tot_rows):
        if ctr not in filtered_rows:
            filtered_data[filt_ctr, :] = in_dict['data'][ctr, :]
            filtered_labels[filt_ctr] = in_dict['labels'][ctr]
            filtered_filenames[filt_ctr] = in_dict['filenames'][ctr]
            filt_ctr += 1
    filtered_dict['data'] = filtered_data
    filtered_dict['labels'] = filtered_labels
    filtered_dict['filenames'] = filtered_filenames
    filtered_dict['batch_label'] = in_dict['batch_label']

    return filtered_dict


overbal_classes = [2,3,4,5]
prob_of_elim_non_over = 0.33
output_file = 'bcdd_1_23'
main_data = 'data_batch_1'
extra_data = ['data_batch_2', 'data_batch_3']#, 'data_batch_4', 'data_batch_5']


od = get_overbalanced_classes(overbal_classes, extra_data)
out_dict = make_unbalanced_dict(main_data, od, output_file)
out_dict = filter_classes(out_dict, overbal_classes, prob_of_elim_non_over)
cPickle.dump(out_dict, open(os.path.join(default_path,
                                         out_dict['batch_label']),
                            'wb'))
