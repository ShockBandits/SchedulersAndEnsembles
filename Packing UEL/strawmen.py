from collections import defaultdict
import numpy as np
import random
import sys
sys.path.append('/home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL')


class StrawMan1(object):

    def __init__(self, ensemble, subset_size):
        self.ensemble = ensemble
        self.subset_size = subset_size
        self.queue = []

    def newArrival(self, new_image_list):
        self.queue += new_image_list

    def schedule(self):
        if len(self.queue) == 0:
            return []
        num_scheduled_images = min(self.subset_size, len(self.queue))
        sample_image_nums = self.queue[:num_scheduled_images]
        self.queue = self.queue[num_scheduled_images:]

        tot_num_classifiers = self.ensemble.classifier_instance_list

        output_list = []
        done = False
        curr_image_ctr = 0

        # Start processing images
        while not done:
            eligible_classifiers = tot_num_classifiers
            random.shuffle(eligible_classifiers)

            while len(eligible_classifiers) > 0:

                # Get next image
                curr_image_num = sample_image_nums[curr_image_ctr]
                # print "Current Image Counter: %d" % curr_image_ctr

                # Get subset of classifiers to be used for current image
                num_chosen_classifiers = min(self.subset_size, len(eligible_classifiers))
                chosen_classifiers = eligible_classifiers[:num_chosen_classifiers]
                eligible_classifiers = eligible_classifiers[num_chosen_classifiers:]

                # Prepare to record results
                vote_dict = defaultdict(int)
                clfr_votes = dict()

                # Let each classifier in chosen subset classify image
                for ctr, clfr in enumerate(chosen_classifiers):
                    samp = clfr.get_sample(clfr.test_data,
                                           curr_image_num)
                    labels = clfr.classify(samp)
                    most_prob_label = np.argmax(labels)
                    vote_dict[most_prob_label] += 1
                    clfr_votes[clfr.name] = most_prob_label + 1
                    
                # Record if current classifiers voted correctly or not
                answers = [x for x in vote_dict if vote_dict[x] == max(vote_dict.values())]
                random.shuffle(answers)  # ensuring random pick in case of tie
                output_list.append((curr_image_num, answers[0]+1, clfr_votes))

                # Prepare to process next image
                curr_image_ctr += 1
                if curr_image_ctr >= len(sample_image_nums):
                    eligible_classifiers = []
                    done = True

        return output_list
