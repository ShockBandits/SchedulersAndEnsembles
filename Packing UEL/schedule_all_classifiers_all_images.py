from collections import defaultdict
import numpy as np
import random
import sys
sys.path.append('/home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL')

from Ensembles.Ensemble import Ensemble
x = Ensemble('./Cifar10/',  3)
x.load_classifiers()
x.get_train_data()
x.get_test_data()

x.assign_members_train_data()
x.assign_members_test_data()

x.classifier_list = []
for curr in x.classifier_dict:
    x.classifier_list += x.classifier_dict[curr]


def rnd_subsets_test(self, sample_image_nums, subset_size):
    """Submit images to randomly chosen subsets of
    size = min(subset_size,
               number available classifiers in time slot)"""

    # Initialize administrative variables
    results_dict = {'right': 0.0, 'wrong': 0.0}
    done = False
    time_slot = 0
    curr_image_ctr = 0

    # Start processing images
    while not done:
        # Initialize variables for new time slot
        time_slot += 1
        eligible_classifiers = self.classifier_list
        random.shuffle(eligible_classifiers)

        while len(eligible_classifiers) > 0:

            # Get nest image
            curr_image_num = sample_image_nums[curr_image_ctr]
            print "Current Image Counter: %d" % curr_image_ctr

            # Get subset of classifiers to be used for current image
            num_chosen_classifiers = min(subset_size, len(eligible_classifiers))
            chosen_classifiers = eligible_classifiers[:num_chosen_classifiers]
            eligible_classifiers = eligible_classifiers[num_chosen_classifiers:]

            # Prepare to record results
            vote_dict = defaultdict(int)

            # Let each classifier in chosen subset classify image
            for ctr, clfr in enumerate(chosen_classifiers):
                samp = clfr.get_sample(clfr.test_data,
                                       curr_image_num)
                labels = clfr.classify(samp)
                most_prob_label = np.argmax(labels)
                vote_dict[most_prob_label] += 1

            # Record if current classifiers voted correctly or not
            answers = [x for x in vote_dict if vote_dict[x] == max(vote_dict.values())]
            if self.test_labels[curr_image_num] not in answers:
                results_dict['wrong'] += 1
            else:
                results_dict['right'] += 1.0/len(answers)

            # Prepare to process next image
            curr_image_ctr += 1
            if curr_image_ctr >= len(sample_image_nums):
                eligible_classifiers = []
                done = True

    # Get final results and return
    results_dict['total_time_slots'] = time_slot
    results_dict['acc'] = results_dict['right'] / (results_dict['right'] +
                                                   results_dict['wrong'])
    return results_dict



def slow_test(self, sample_image_nums):

    results_dict = {'right': 0.0, 'wrong': 0.0}
    for curr_image_num in sample_image_nums:
        print "Current Image Number: %d" % curr_image_num
        # print "Current Image Label: %d" % self.test_labels[curr_image_num]
        vote_dict = defaultdict(int)

        for curr_clfr_type in self.classifier_dict:
            for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                samp = clfr.get_sample(clfr.test_data,
                                       curr_image_num)
                labels = clfr.classify(samp)
                most_prob_label = np.argmax(labels)
                vote_dict[most_prob_label] += 1

        print "---------------------------------------------------\n"
        answers = [x for x in vote_dict if vote_dict[x] == max(vote_dict.values())]
        if self.test_labels[curr_image_num] not in answers:
            results_dict['wrong'] += 1
            # print "Wrong"
        else:
            results_dict['right'] += 1.0/len(answers)
            # print "Right", 1.0/len(answers)
    results_dict['acc'] = results_dict['right'] / (results_dict['right'] +
                                                   results_dict['wrong'])
    return results_dict


# print slow_test(x, range(len(x.test_labels)))
# Results
# {'acc': 0.8168759755562583, 'wrong': 1795.0, 'right': 8007.0999999999985}


x.rnd_subsets_test(range(len(x.test_labels)), 1)
#print rnd_subsets_test(x, range(len(x.test_labels)), 1)
# {'acc': 0.7443, 'wrong': 2557.0, 'right': 7443.0, 'total_time_slots': 2000}

#print rnd_subsets_test(x, range(len(x.test_labels)), 2)
#{'acc': 0.8046626823884133, 'wrong': 1814.0, 'right': 7472.5, 'total_time_slots': 3334}

#print rnd_subsets_test(x, range(len(x.test_labels)), 3)
#{'acc': 0.8264751761605074, 'wrong': 1613.0, 'right': 7682.499999999976, 'total_time_slots': 5000}

#print rnd_subsets_test(x, range(len(x.test_labels)), 4)
#{'acc': 0.788261781712122, 'wrong': 2069.0, 'right': 7702.5, 'total_time_slots': 5000}

#print rnd_subsets_test(x, range(len(x.test_labels)), 5)
#{'acc': 0.8168759755562583, 'wrong': 1795.0, 'right': 8007.0999999999985, 'total_time_slots': 10000}
