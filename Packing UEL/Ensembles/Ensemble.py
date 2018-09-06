from collections import defaultdict
from configobj import ConfigObj, ConfigObjError
import importlib
import numpy as np
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'Cifar10', 'Accessor'))
from readCifar10 import *
# need to change the above


class Ensemble(object):

    def __init__(self, data_set, ens_num=0):
        
        # load the config file
        home_dir = os.path.dirname(__file__)
        cfg_file = os.path.join(home_dir, data_set, 'cfg_dir',
                                'ensemble_'+str(ens_num)+'.cfg')
        
        try:
            config = ConfigObj(cfg_file, file_error=True)
        except (ConfigObjError, IOError), e:
            print "\n\nCouldn't read '%s' : %s\n\n" % (cfg_file, e)
            sys.exit(1)
            
        self.classifier_dir = config['classifier_root_dir']
        self.classifier_dict = defaultdict(list)
        self.classifier_list = []
        
        # the classifiers used in the ensemble
        clfr_dict = config['classifiers']
        import_package = 'Cifar10.Classifiers.'
        self.classifier_instance_list = []
        for curr_clfr in clfr_dict:
            import_clfr = import_package + curr_clfr
            try:
                c_module = importlib.import_module(import_clfr)
            except:
                c_module = importlib.import_module("Ensembles." + import_clfr)
            constructor = getattr(c_module, curr_clfr+'_Classifier', None)
            
            self.classifier_list.append(curr_clfr)
            
            if isinstance(clfr_dict[curr_clfr], str):
                self.classifier_dict[curr_clfr].append(constructor(clfr_dict[curr_clfr]))
                print "Instantiated %s - %s" % (curr_clfr,
                                                clfr_dict[curr_clfr])
            elif isinstance(clfr_dict[curr_clfr], list):
                for clfr_elem in clfr_dict[curr_clfr]:
                    self.classifier_dict[curr_clfr].append(constructor(clfr_elem))
                    print "Instantiated %s - %s" % (curr_clfr,
                                                    clfr_elem)
            else:
                print "\n\nCan't Read Config Info For %s -  %s\n\n" % (curr_clfr,
                                                                       clfr_dict[curr_clfr])
                sys.exit(1)
        print "\n"

        self.has_train_data = False
        self.has_test_data = False
        self.train_data_file = config['datasets']['train_data_file']
        self.test_data_file = config['datasets']['test_data_file']
        self.classifier_list.sort()
        self.classifier_instance_list = []
        for curr in self.classifier_dict:
            self.classifier_instance_list += self.classifier_dict[curr]

    def get_train_data(self):
        cifar_dict = getCifar10(self.train_data_file)
        self.train_data = cifar_dict['data']
        self.train_labels = np.asarray(cifar_dict['labels'])
        self.has_train_data = True
    
    def get_test_data(self):
        cifar_dict = getCifar10(self.test_data_file)
        self.test_data = cifar_dict['data']
        self.test_labels = np.asarray(cifar_dict['labels'])
        self.has_test_data = True
    
    # -----------------------------------------------
    #         Classifier instantiation functions
    # -----------------------------------------------
    def assign_members_train_data(self):
        print 'Getting Train Data'
        for curr_clfr_type in self.classifier_dict:
            for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                print "Geting data for", curr_clfr_type
                clfr.get_train_data(self.train_data_file)
        print

    def assign_members_test_data(self):
        print 'Getting Test Data'
        for curr_clfr_type in self.classifier_dict:
            for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                print "Geting data for", curr_clfr_type
                clfr.get_test_data(self.test_data_file)
        print
    
    def create_classifiers(self):
        print 'Creating Classifiers'
        for curr_clfrs in self.classifier_dict:
            for curr in self.classifier_dict[curr_clfrs]:
                curr.create()
        print "\n"
        
    def save_classifiers(self):
        print 'Saving Classifiers'
        for curr_clfrs in self.classifier_dict:
            for curr in self.classifier_dict[curr_clfrs]:
                curr.save()
        print "\n"
        
    def fit_classifiers(self):
        print 'Fitting Classifiers'
        for curr_clfrs in self.classifier_dict:
            for curr in self.classifier_dict[curr_clfrs]:
                curr.fit()
        print "\n"
    
    def load_classifiers(self):
        print 'Loading Classifiers'
        for curr_clfrs in self.classifier_dict:
            for curr in self.classifier_dict[curr_clfrs]:
                curr.read()
        print "\n"

    def print_indiv_acc(self, classifier_type, classifier_name, classifier_num=0):
        classifier = self.classifier_dict[classifier_type][classifier_num]
        print "%s - Classifier %s (%s):" % (classifier_type, classifier_num, classifier_name)
        classifier.print_acc()

    def print_all_accs(self):
        for curr_clfr_type in self.classifier_dict:
            for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                self.print_indiv_acc(curr_clfr_type, clfr.name, ctr)
                print 
            print 
        print 

    def test_classifiers(self, sample_image_nums):
        for curr_image_num in sample_image_nums:
            print "Current Image Number: %d" % curr_image_num
            print "Current Image Label: %d" % self.train_labels[curr_image_num]

            for curr_clfr_type in self.classifier_dict:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    samp = clfr.get_sample(clfr.train_data,
                                           curr_image_num)
                    labels = clfr.classify(samp)
                    print clfr.name, " - Most Probable Label:", np.argmax(labels)
                    print labels, '\n'
            print "---------------------------------------------------\n"

    def get_confidence_arrays(self, sample_image_nums):

        out_dict = dict()
        num_classes = len(set(self.train_labels))
        for curr_image_num in sample_image_nums:

            curr_image_label = self.train_labels[curr_image_num]
            out_dict[curr_image_num] = dict()
            temp = np.zeros([num_classes])
            temp[curr_image_label] = 1.0
            out_dict[curr_image_num]['Oracle'] = temp
            
            for curr_clfr_type in self.classifier_dict:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    samp = clfr.get_sample(clfr.train_data,
                                           curr_image_num)
                    labels = clfr.classify(samp)
                    out_dict[curr_image_num][clfr.name] = labels
        return out_dict

    # --------------------------------------------------
    #    Methods to interface with the estimator
    # --------------------------------------------------
    # All the data is specfic to self.test_data

    def get_p_true(self):
        num_data = len(self.test_labels)
        if self.has_test_data:
            return np.bincount(self.test_labels)/float(num_data)
        else:
            print 'Test data not loaded'
            return None

    def get_conf_matrix(self):
        conf_matrices = {}
        count = 0
        print "Getting conf_matrix"
        if self.has_test_data:
            for curr_clfr_type in self.classifier_list:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    print "Getting conf_matrix for %s-%s" % (str(curr_clfr_type), str(ctr))
                    conf_matrices[count] = clfr.get_conf_matrix(clfr.test_data,
                                                                clfr.test_labels)
                    count += 1
            return conf_matrices
        else:
            print 'Test data not loaded'
            return None
    # --------------------------------------------------
    #    Methods to interface with the scheduler
    # --------------------------------------------------
    # All the data is specific to self.test_data
    
    def classify(self, sample_image_nums, disp=False):
        # returns the classification w.r.t. the sample image numbers
        # -1 represents No class 
        len_samples = len(sample_image_nums)
        i = 0
        label = np.ones(len_samples)*(-1)
        
        if disp: 
            print "Schedule:", sample_image_nums
            
        if self.has_test_data:
            for curr_clfr_type in self.classifier_list:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    curr_image_num = int(sample_image_nums[i]-1)
                    if curr_image_num >= 0:
                        samp = clfr.get_sample(clfr.test_data,
                                               curr_image_num)
                        clfr_labels = clfr.classify(samp)
                        label[i] = np.argmax(clfr_labels)+1 
                        if disp:
                            print "Image %d -> Classifier%s-%s" % (curr_image_num,
                                                                   str(curr_clfr_type),
                                                                   str(ctr))
                            print "Label prob: ", clfr_labels
                            print "Label out:", label[i]
                            print
                    i += 1
        else:
            print 'Test data not loaded'
        return label
    
    # --------------------------------------------------
    #    Methods to interface with the tester
    # --------------------------------------------------
    def import_name_classifier(self):
        name_list = []
        for curr_clfr_type in self.classifier_list:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    name_list += [str(curr_clfr_type)+'-'+str(ctr)]
        return name_list
    
    def import_num_classifier(self):
        return np.sum([len(x) for x in self.classifier_dict.values()])
        
    def import_num_class(self):
        # this is for cifar10 dataset change ir
        return len(getMetaDict()['label_names'])
    
    def import_labels(self):
        if self.has_test_data:
            return self.test_labels
        else:
            print 'Test data not loaded'
            return None
        
    def return_real_label(self, sample_image_nums):
        result = []
        if self.has_test_data:
            for curr_image_num in sample_image_nums:
                result += self.test_labels[curr_image_num]
            return result
        else:
            print 'Test data not loaded'
            return None

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
            eligible_classifiers = self.classifier_instance_list
            random.shuffle(eligible_classifiers)

            while len(eligible_classifiers) > 0:

                # Get next image
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
