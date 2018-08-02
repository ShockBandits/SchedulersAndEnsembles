from collections import defaultdict
import cPickle as pickle
from configobj import ConfigObj, ConfigObjError, flatten_errors
import importlib
import numpy as np
import os
import pandas as pd
import sys
from validate import Validator

sys.path.append(os.path.join(os.path.dirname(__file__), 'Cifar10', 'Accessor'))
from readCifar10 import *
# need to change the above


class Ensemble(object):

    def __init__(self, data_set, ens_num = 0):
        
        # load the config file
        home_dir = os.path.dirname(__file__)
        cfg_file = os.path.join(home_dir, data_set, 'cfg_dir','ensemble_'+str(ens_num)+'.cfg')
        
        try:
            config = ConfigObj(cfg_file, file_error = True)
        except (ConfigObjError, IOError), e:
            print "\n\nCouldn't read '%s' : %s\n\n"%(cfg_file, e)
            sys.exit(1)
            
        self.classifier_dir = config['classifier_root_dir']
        self.classifier_dict = defaultdict(list)
        self.classifier_list = []
        
        # the classifiers used in the ensemble
        clfr_dict = config['classifiers']
        import_package = 'Cifar10.Classifiers.'
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
                print "Instantiated %s - %s"%(curr_clfr,
                                                        clfr_dict[curr_clfr])
            elif isinstance(clfr_dict[curr_clfr], list):
                for clfr_elem in clfr_dict[curr_clfr]:
                    self.classifier_dict[curr_clfr].append(constructor(clfr_elem))
                    print "Instantiated %s - %s"%(curr_clfr,
                                                            clfr_elem)
            else:
                print "\n\nCan't Read Config Info For %s -  %s\n\n"%(curr_clfr,
                                                                     clfr_dict[curr_clfr])
                sys.exit(1)
        print "\n"

        self.has_train_data = False
        self.has_test_data = False
        self.train_data_file = config['datasets']['train_data_file']
        self.test_data_file = config['datasets']['test_data_file']
        self.classifier_list.sort() 

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
    
    #-----------------------------------------------
    #         Classifier instatiation functions
    #-----------------------------------------------
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
    #-----------------------------------------------

    def print_indiv_acc(self, classifier_type, classifier_name, classifier_num = 0):
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

    def test_classifiers(self,sample_image_nums):
        for curr_image_num in sample_image_nums:
            print "Current Image Number: %d"%(curr_image_num)  
            print "Current Image Label: %d"%(self.train_labels[curr_image_num])

            for curr_clfr_type in self.classifier_dict:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    samp = clfr.get_sample(clfr.train_data,
                                           curr_image_num)
                    labels = clfr.classify(samp)
                    print clfr.name," - Most Probable Label:", np.argmax(labels)
                    print labels,'\n'
            print "---------------------------------------------------\n"
    #--------------------------------------------------
    #    Methods to interface with the estimator
    #--------------------------------------------------
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
        print "Geting conf_matrix"
        if self.has_test_data:
            for curr_clfr_type in self.classifier_list:
                for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                    print "Geting conf_matrix for %s-%s"%(str(curr_clfr_type), str(ctr))
                    conf_matrices[count] = clfr.get_conf_matrix(clfr.test_data,
                                                                clfr.test_labels)
                    count +=1
            return conf_matrices
        else:
            print 'Test data not loaded'
            return None
    #--------------------------------------------------
    #    Methods to interface with the scheduler
    #--------------------------------------------------
    # All the data is specfic to self.test_data       
    
    def classify(self, sample_image_nums, disp = False):
        # returns the classification w.r.t. the sample image numbers
        # -1 represents No class 
        len_samples = len(sample_image_nums);
        len_clfr_list = len(self.classifier_list)
        i = 0;
        label = np.ones(len_samples)*(-1)
        
        if disp: 
            print "Schedule:",sample_image_nums
            
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
                            print "Image %d -> Classifier%s-%s"%(curr_image_num,
                                                                 str(curr_clfr_type), str(ctr))
                            print "Label prob: ", clfr_labels
                            print "Label out:", label[i]
                            print
                    i+=1;
        else:
            print 'Test data not loaded'
        return label
    
    #--------------------------------------------------
    #    Methods to interface with the tester
    #--------------------------------------------------
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
        
    
                
                
            
        
            

        
                                                        
