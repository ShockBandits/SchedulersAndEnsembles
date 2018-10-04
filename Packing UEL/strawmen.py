from collections import defaultdict
import math
import numpy as np
import random
import sys
sys.path.append('/home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL')


class StrawMan1(object):

    def __init__(self, ensemble, subset_size):
        self.ensemble = ensemble
        self.subset_size = subset_size
        self.queue = []
        self.ensemble_size = len(ensemble.classifier_instance_list)
        if self.ensemble_size < subset_size:
            sys.exit("Too few classifier in ensemble for that subset size")
        self.max_images = math.ceil(float(self.ensemble_size)/ subset_size)
        self.tot_images_seen = 0
        self.accuracy_tally = [(0, 0, 0, 0, 0, self.tot_images_seen)]

    def newArrival(self, new_image_list):
        """Adds new images to processing queue"""
        self.queue += new_image_list

    def schedule(self):
        """ Sends images to be classified. Each image is sent to a
             number of classifiers equal to the subset_size, or
             how ever many are available. The return list contains
             for each image, an identifying number, the consensus
             classification of the classifiers and a dictionary
             listing the decisions of the individual classifiers"""
        if len(self.queue) == 0:
            return []
        num_scheduled_images = int(min(self.max_images, len(self.queue)))
        sample_image_nums = self.queue[:num_scheduled_images]
        self.queue = self.queue[num_scheduled_images:]

        tot_num_classifiers = self.ensemble.classifier_instance_list

        output_list = []
        done = False
        curr_image_ctr = 0
        num_right_abs = num_right_partial = num_partial = num_wrong = 0

        eligible_classifiers = tot_num_classifiers
        random.shuffle(eligible_classifiers)

        # Start processing images - process until run out of images
        # or classifiers
        while not done:

            # Get next image
            curr_image_num = sample_image_nums.pop() #sample_image_nums[curr_image_ctr]
            #print "Current Image Num & Counter: %d -- %d" % (curr_image_num, curr_image_ctr)

            # Get subset of classifiers to be used for current image
            num_chosen_classifiers = min(self.subset_size,
                                         len(eligible_classifiers))
            chosen_classifiers = eligible_classifiers[:num_chosen_classifiers]
            eligible_classifiers = eligible_classifiers[num_chosen_classifiers:]

            # Prepare to record results
            vote_dict = defaultdict(int)
            clfr_votes = dict()
            true_labels_list = []

            # Let each classifier in chosen subset classify image
            for ctr, clfr in enumerate(chosen_classifiers):
                samp = clfr.get_sample(clfr.test_data,
                                       curr_image_num)
                true_label = clfr.get_sample_label(clfr.test_labels,
                                                   curr_image_num)

                true_labels_list.append(true_label)
                labels = clfr.classify(samp)
                most_prob_label = np.argmax(labels)
                vote_dict[most_prob_label] += 1
                clfr_votes[clfr.name] = most_prob_label + 1

            # Record if current classifiers voted correctly or not
            if len(set(true_labels_list)) != 1:
                print ("Images Misaligned")
                sys.exit()
            answers = [x for x in vote_dict if vote_dict[x] == max(vote_dict.values())]
            random.shuffle(answers)  # ensuring random pick in case of tie
            output_list.append((curr_image_num, answers[0]+1, clfr_votes))
            if true_labels_list[0] in answers and len(answers) > 1:
                num_right_partial += 1./float(len(answers))
                num_partial += 1
            elif answers[0] == true_labels_list[0] and len(answers) == 1:
                num_right_abs += 1
            else:
                num_wrong += 1
            self.tot_images_seen += 1

            # Prepare to process next image
            curr_image_ctr += 1
            if len(sample_image_nums) < 1 or len(eligible_classifiers) < self.subset_size: #1:
            #if curr_image_ctr >= len(sample_image_nums) or len(eligible_classifiers) < self.subset_size:  # 1:
                    done = True

        # Update accuracy tally
        self.queue += sample_image_nums
        curr_tuple = self.accuracy_tally[-1]
        if len(self.accuracy_tally) != curr_tuple[0] + 1:
            sys.exit("Error in time-stamping of accuracy_tally")
        else:
            time_stamp = curr_tuple[0] + 1
            tot_right_abs = curr_tuple[1] + num_right_abs
            tot_right_partial = curr_tuple[2] + num_right_partial
            tot_partial = curr_tuple[3] + num_partial
            tot_wrong = curr_tuple[4] + num_wrong
            self.accuracy_tally.append((time_stamp, tot_right_abs,
                                        tot_right_partial, tot_partial,
                                        tot_wrong, self.tot_images_seen))
        return output_list
