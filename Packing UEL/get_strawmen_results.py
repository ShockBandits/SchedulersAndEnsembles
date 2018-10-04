from collections import defaultdict
import numpy as np
import os
import pickle
import random
import sys

sys.path.append('/home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL')

_full_path = os.path.realpath(__file__)
_dir_path, _filename = os.path.split(_full_path)
results_path = os.path.join(_dir_path, 'results_dir')
_cfg_path = os.path.join(_dir_path, 'cfg_dir')

from Ensembles.Ensemble import Ensemble
from strawmen import StrawMan1


def make_ensemble(cfg_file_num):
    # Create Ensemble
    x = Ensemble('./Cifar10/', cfg_file_num)
    x.load_classifiers()
    x.get_train_data()
    x.get_test_data()

    x.assign_members_train_data()
    x.assign_members_test_data()

    return x


def defaultdict_func():
    return {'pure_acc': list(), 'exp_acc': list(),
            'orac_acc': list(), 'time_slots': list()}


def run_simulation(ensemble, subset_size, max_new_arrivals):
    """ Tests strawman scheduler"""

    # Create strawman
    strawman = StrawMan1(ensemble, subset_size)

    sample_list = range(strawman.ensemble.test_data.shape[0])
    random.shuffle(sample_list)
    done = False
    detailed_dict = dict()

    while not done:
        # Calculate new arrivals to queue
        if len(sample_list) != 0:
            num_new_arrivals = random.randint(0, max_new_arrivals)
            num_new_arrivals = min(len(sample_list),
                                   num_new_arrivals)
            curr_samps = sample_list[:num_new_arrivals]
            sample_list = sample_list[num_new_arrivals:]
            strawman.newArrival(curr_samps)

        # Let strawman do single time step of processing
        results = strawman.schedule()
        for curr in results:
            detailed_dict[curr[0]] = curr

        if len(sample_list) == 0 and len(strawman.queue) == 0:
            done = True
    final_results = strawman.accuracy_tally[-1]
    return (final_results, detailed_dict)


def run_batch(output_file, num_simulations=5,
              max_subset_size=5, max_arrivals=10,
              ensemble_id_num=7):
    # Test it
    ensemble = make_ensemble(ensemble_id_num)
    result_dict = defaultdict(defaultdict_func)
    stats_dict = defaultdict(defaultdict_func)
    all_details_dict = defaultdict(list)

    for _ in range(num_simulations):

        for ctr in range(1, max_subset_size + 1):
            answers, detailed_dict = run_simulation(ensemble, ctr, max_arrivals)
            all_details_dict[ctr].append(detailed_dict)

            print "-------------------------------------------------------------------"
            print "Subset Size: %d  Time Slots: %d  Tot Images: %d" % (ctr, answers[0], answers[5])
            pure_acc = float(answers[1]) / answers[5]
            exp_acc = float(answers[1] + answers[2]) / answers[5]
            orac_acc = float(answers[1] + answers[3]) / answers[5]
            print "     Pure_Acc: %6.4f   Expected Acc: %6.4f Oracular Acc: %6.4f" % (pure_acc, exp_acc, orac_acc)
            result_dict[ctr]['pure_acc'].append(pure_acc)
            result_dict[ctr]['exp_acc'].append(exp_acc)
            result_dict[ctr]['orac_acc'].append(orac_acc)
            result_dict[ctr]['time_slots'].append(answers[0])

    print "-------------------------------------------------------------------"

    out_fields = ['pure_acc', 'exp_acc', 'orac_acc', 'time_slots']
    out_str = ''
    for ctr in range(1, max_subset_size + 1):
        out_str += "========================================================================\n"
        out_str += "Subset Size: %d  Tot Images: %d\n" % (ctr, ensemble.test_data.shape[0])
        for curr_field in out_fields:
            stats_dict[ctr][curr_field] = (np.average(result_dict[ctr][curr_field]),
                                           np.std(result_dict[ctr][curr_field]))

            out_str += "   %s  Avg: %6.4f  Std: %6.4f\n" % (curr_field,
                                                            stats_dict[ctr][curr_field][0],
                                                            stats_dict[ctr][curr_field][1])
    out_str += "========================================================================\n"
    print out_str

    with open(os.path.join(results_path, output_file + '.txt'), 'wb') as f:
        f.write(out_str)
    pickle.dump((stats_dict, result_dict, all_details_dict),
                open(os.path.join(results_path, output_file + '.pkl'), 'wb'))


run_batch("Strawman_Test5")
