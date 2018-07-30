import numpy as np
from itertools import product
import cvxpy as cvx
import traceback
import itertools
from matplotlib import pyplot as plt
from scipy.stats import entropy

from Ensembles.Ensemble import Ensemble
from Ensembles.Cifar10.Accessor.readCifar10 import *

import import_ipynb
from spectralEM import spectralEM
from conflictGraph import conflictGraph
from fakeENS import fakeENS



def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = np.zeros(n)
        for bit in bits:
            s[bit] = 1
        result.append(s)
    return result

def str_arr(arr, mode = 'float'):
    if mode == 'int':
        return " ".join("%2d"%x for x in arr)
    else:
        return " ".join("%.3f"%x for x in arr)
def str_mat(mat):
    return "\n".join(str_arr(arr) for arr in mat) 

def averageAcc(p_true, C):
    return np.sum(C[i,i]*p for i, p in enumerate(p_true))    

def computeError(output_stream, true_label_stream):
    if len(output_stream) == 0:
        return 0
    est_label = [o[1] for o in output_stream]
    error = [(np.abs(x-y)>0.5) for x,y in zip(est_label, true_label_stream)]
    error_frac = (np.sum(error)/float(len(error)))
    return error_frac


def ConfError(C_real, num_data, p_true, num_classifier, C_est, ph_est_avg):
    errp = np.linalg.norm(p_true - ph_est_avg, 1)
    err = [np.linalg.norm(C_real[k]- C_est[k], 1) for k in range(num_classifier)]
    print '---------------------------'
    print '*****Param Update*****'
    print '---------------------------'
    print 'Num samples:%d'%num_data
    print 'True ph:%s'%str_arr(p_true)
    print 'Est ph:%s'%str_arr(ph_est_avg)
    print 'L1 error in class probability:%.3f'%errp
    print 'Sum of L1 error in conf matrices:%.3f'%np.sum(err)
    print '---------------------------'
    return errp, err

def simulator(data_set = 'Cifar10',
              num_class = 3, num_classifier = 5,  # Ensemble and Data
              load_ENS = False, fit_ENS = False, 
              load_params = True, save_params = False,# Ensemble
              k_max_size = 3, # allowed subset 
              arr_rate = 0.8, thres = 0.75, V = 1, # arrival and departure thres
              updateON = False, truePrior = True, # priors
              time_slots = 1000 # simulation length
             ):
    '''
    The function that simulates the system with the input parameters
    
    '''
    #---------------------------------
    #       ENS and Data
    #---------------------------------
    if load_ENS:
        ENS=Ensemble(data_set)

        # fit the ENS again
        if fit_ENS:
            ENS.get_train_data(); ENS.assign_members_train_data()
            ENS.get_test_data(); ENS.assign_members_test_data()
            ENS.create_classifiers(); ENS.fit_classifiers()
            ENS.save_classifiers()

        '''    
        try:
            # fit the ENS again
            if fit_ENS:
                ENS.get_train_data(); ENS.assign_members_train_data()
                ENS.get_test_data(); ENS.assign_members_test_data()
                ENS.create_classifiers(); ENS.fit_classifiers()
                ENS.save_classifiers()
            ENS.load_classifiers()
        except:
            print 'Error in loading classifiers'
            ENS.get_train_data(); ENS.assign_members_train_data()
            ENS.get_test_data(); ENS.assign_members_test_data()
            ENS.create_classifiers(); ENS.fit_classifiers()
            ENS.save_classifiers()
        '''

        ENS.load_classifiers()
        ENS.get_train_data(); ENS.assign_members_train_data()
        ENS.get_test_data(); ENS.assign_members_test_data()
        trueConfMat = ENS.get_conf_matrix()
        
        # info about classifiers
        name_classifier = ENS.import_name_classifier()
        num_classifier = ENS.import_num_classifier()
        num_class = ENS.import_num_class()
        
        # obtain the samples
        true_labels = ENS.import_labels()
        num_tot_samp_stream = len(true_labels)
        samples = np.arange(num_tot_samp_stream)
        
        # permute the samples and the labels
        perm_data = np.random.permutation(num_tot_samp_stream)
        stream_true_labels = true_labels[perm_data] # Begins from 0
        stream_sample = samples[perm_data] # Begins from 0
        
        # create arrival array and change time_slots if sample is less
        arrival_arr = np.random.poisson(arr_rate, time_slots)
        if np.sum(arrival_arr) > num_tot_samp_stream: 
            time_slots = np.argmax(np.cumsum(arrival_arr)> num_tot_samp_stream)-1
        num_tot_samp = np.sum(arrival_arr[:time_slots])
        
        print 'Number of total samples in stream:%d'%num_tot_samp_stream
        print 'Number of timeslots:%d'%time_slots
        print 'Number of arrivals:%d'%num_tot_samp
        print
        
        # obtain the parameters (w.r.t. test data)
        if load_params:
            trueConfMat = np.load('conf_mat.npy').item()
            p_true = np.load('p_true.npy')
        else:
            trueConfMat = ENS.get_conf_matrix()
            p_true = ENS.get_p_true()
    else:
        if load_params:
            conf_mat = np.load('conf_mat.npy').item() 
            p_true = np.load('p_true.npy')
            #print p_true
            #print conf_mat.items()
        else:
            # probability of the various classes
            p_const = 0.5
            p_true = np.random.rand(num_class); 
            p_true = (p_const+p_true)/ np.sum(p_const+p_true)
            conf_mat = None
        #-----------------------------------
        # synthetic dataset
        name_classifier = ['syn-'+str(i) for i in range(num_classifier)]
        ENS = fakeENS(num_classifier, num_class, confMat = conf_mat)
        trueConfMat = ENS.getConfMat()
        #------------------------------------
        # beginning of the simulation
        # can be restarted to rerun the simulation
        # rerunning following blocks(including this one)
        # do not change the capacity region
        arrival_arr = np.random.poisson(arr_rate, time_slots)
        num_tot_samp = np.sum(arrival_arr) 

        # generate the samples(both begins from 0)
        stream_sample = np.arange(num_tot_samp) 
        stream_true_labels = np.random.choice(num_class,
                                              num_tot_samp, p = p_true)
        
        # feed samples to the ensemble
        ENS.newSamples(stream_true_labels) 
        # re-randomize the output label mapping
        ENS.reshuffle() 
    #--------------------------------------------------------
    #.         Display Ensemble and Data
    #-------------------------------------------------------- 
    if save_params:
        np.save('conf_mat.npy', trueConfMat)
        np.save('p_true.npy', p_true)
    #-----------------------------------
    print "Number of class:%d"%num_class
    print "Number of classifiers:%d"%num_classifier
    print "Name of classifiers:",name_classifier
    print 'True label probability:', p_true
    print 'True conf matrices'
    for i, mat in enumerate(trueConfMat.values()): 
        print 'Classifier:%s'%name_classifier[i]
        print str_mat(mat)
        print
    #--------------------------------------------------------
    #.         Conflict Graph and Spectral Estimator
    #--------------------------------------------------------
    # init params for spectral estimator and the conflict graph
    params = {}
    unip = np.ones(num_class)/float(num_class);
    # the initial conf matries
    expert_frac = 0.01
    expertConfMat = {j: (expert_frac)*np.eye(num_class)+ 
                     (1-expert_frac)*np.tile(unip, [num_class, 1])
                     for j in range(num_classifier)}
    if truePrior:
        # True prior
        params['Confmat'] = trueConfMat
        params['p_true'] = p_true
    else:
        # expert prior with uniform label porbabilities
        params['Confmat'] = expertConfMat
        params['p_true'] = unip
    #---------display: current parameter error--------------
    est_err = ConfError(trueConfMat, 0, p_true, num_classifier, 
                         params['Confmat'], params['p_true'])
    
    true_avg_acc = [averageAcc(p_true, trueConfMat[i]) for i in range(num_classifier)]
    init_avg_acc = [averageAcc(p_true, params['Confmat'][i]) for i in range(num_classifier)]
    print 'True Avg accuracy: %s'%str_arr(true_avg_acc)
    print 'Initial Avg accuracy: %s'%str_arr([averageAcc(p_true, params['Confmat'][i]) 
                                              for i in range(num_classifier)])
    
    #                Allowed subsets
    
    k_max_size = min(num_classifier-1, k_max_size)
    k_max = [num_classifier]
    for kms in range(k_max_size):
        k_max.append(int(kms+1))
        k_max.append(num_classifier - int(kms+1))
    k_max = list(set(k_max))
    
    allowed_subset = []
    for k in k_max:
        allowed_subset += kbits(int(num_classifier), int(k))
    
    #           Create the conflict graph
    
    disp = 100    # disp parameter for conflict Graph
    G = conflictGraph(allowed_subset = allowed_subset, M = num_classifier, 
                      C = num_class, Ens= ENS, params = params, 
                      V = V, thres = thres, disp = disp)
    
    #          Create the spectral estimator
    
    S = spectralEM(num_classifier, num_class, maxiter = 500, num_init = 100)
    
    #          Compute metrics for Spectral estimation
    
    group = S.group 
    groupConfMat = {i: np.mean([trueConfMat[j] for j in range(num_classifier) if group[i][j]], axis = 0)  
                    for i in range(3)}

    # Quantifies quality of individual classifiers in confusion matrix
    kappa = np.min([np.min([[groupConfMat[j][l,l] - groupConfMat[j][l,c] for c in range(num_class) if c !=l] 
                          for l in range(num_class)]) for j in range(3)])


    # Quantifies ensemble quality - independence of classifiers
    barD = np.min([[np.mean([entropy(trueConfMat[i][l,:], trueConfMat[j][l,:]) 
                             for i in range(num_classifier)]) 
                    for c in range(num_class) if c !=l] 
                   for l in range(num_class)])
    
    #Display: metrics for spectral estimation
    print 'Metrics in Spectral Learning| kappa:%.3f, barD:%.3f'%(kappa, barD)
    print
    
    #display: allowed subsets
    print 'Allowed all subsets of size:',k_max

    #              Reset Internal States
    
    G.reset() # resets the internal queues
    S.reset() # resets the internal parameters

    #       Exploration and estimation parameters 
    
    update_num = 25 # lenght of explore_data when S is updated
    updateSpectral = True # no spectral updates (only EM)
    # make the following two 0 to stop exploration
    explore_prob_const = 0.1 # probability with which exploration happens
    init_explore = 200 # Number of initial time slots when explore happens
    #---------------------------------------------------
    
    # Initialization
    output_stream= []
    true_label_stream = []
    queue_evol = []
    error_evol = []
    est_evol = []

    samp_count = 0
    tau = 0 
    explore_data = []
    count_update = 0
    
    # display period
    disp_period = 25
    display = 2 # display flag
    
    print 'Arrival Rate:', arr_rate, 'Accuracy Threshold:', G.thres
    print 'Total samples:', num_tot_samp, 'Total time slots:', time_slots
    #--------------------------------------
    #            Iterations
    #--------------------------------------
    print '----------\nIterations Begin\n-----------' 
    while len(output_stream)< num_tot_samp:
        #----arrival-----
        if tau < time_slots:
            # sample id should begin from 1 while providing newArrival
            G.newArrival(stream_sample[samp_count:samp_count+arrival_arr[tau]]+1)
            samp_count += arrival_arr[tau]
        
        #----------------------------------------------------------
        #                  Exploit or Explore: Scheduling
        #----------------------------------------------------------
        
        #                set explore probability 
        
        # default
        explore_prob = explore_prob_const # explore_prob_const*np.log(i)/i
        # explore for the first 100 rounds
        explore_prob = 1.0 if tau < init_explore else explore_prob
        # explore only if updateON
        explore_flag = (np.random.rand() < explore_prob) and updateON
        
        #                 schedule
        
        try:
            sample_out, schedule, _ = G.schedule(explore = explore_flag)
        except Exception as e:
            print '***Error in Scheduling***Exiting***'
            traceback.print_exc(); break

        #----------------------------------------------------------
        #                  Update Parameters
        #----------------------------------------------------------
        if updateON: # update on each output
        #if explore_flag: # update on exploration instances
            explore_data += [s[2]-1 for s in sample_out]
            
        if (len(explore_data) >= update_num) and updateON:
            
            #         update the spectralEM object
            
            S.update(explore_data);
            
            #          recallibrate params using Spectral
            
            if (tau > init_explore) and updateSpectral:
                S.updateParamsSpectral()
                count_update +=1
                
            #           update params using EM
            
            S.updateParamsEM(explore_data);
            
            #           update G params
            
            new_params = {'Confmat': S.conf_mat, 'p_true': S.ph_est_avg}
            G.updateParams(new_params)
            
            #            compute error 
            
            est_err = ConfError(trueConfMat, S.num_data, p_true, 
                           num_classifier, S.conf_mat, S.ph_est_avg)
            
            #           empty explore_data 
            
            explore_data = []
        #----------------------------------------------------------       
        #                    Results
        #----------------------------------------------------------
        true_label_out = []
        for s in sample_out:
            ind_s, = np.where(stream_sample == int(s[0]-1))
            if stream_sample[ind_s] != int(s[0]-1): 
                print "Error in finding sample location"
                true_label_out += [-1]
            else:
                true_label_out += [stream_true_labels[ind_s]+1]
            
        rem_samples = G.totalCount()
        output_stream += sample_out
        tot_queue = G.totalCount()
        true_label_stream += true_label_out
        classification_error = computeError(output_stream, true_label_stream)

        queue_evol.append(tot_queue) 
        error_evol.append(classification_error)
        est_evol.append(est_err)
        #----------------------------------------------------------       
        #                   Display
        #----------------------------------------------------------
        if not np.mod(tau, disp_period): 
            print '--------------------'
            print 'num iterations:', tau;
            print 'Total samples in queues:', G.totalCount()
            print 'Total samples in explore list:', len(explore_data)
            print 'Number of outputs:', len(output_stream)
            print 'Error fraction:%.3f'%classification_error
            print '-------------------'
        if display <= 1:
           
            G.print_labelSubsets()
            print 
        if display <= 2:
            print 'Iteration:', tau, 'Total Queue Length:', tot_queue
            print 'Schedule:', schedule
            print 'Output'
            for i_s, s in enumerate(sample_out):
                # s: sample_id, label_final, label_arr, label_conf
                print " ".join(['Sample Id:%4d'%s[0],
                                'Labels=> Real:%2d'%(true_label_out[i_s]), 
                                'Final:%2d'%s[1], 'Conf:%.3f'%s[3],
                                'All: %s'%str_arr(s[2], 'int')])
            print
                
        #  increase the time count
        
        tau += 1
    #----------------------outputs-----------
    output_dict = {}
    output_dict['tau'] = tau
    output_dict['time_slots'] = time_slots
    output_dict['arrival_arr'] = arrival_arr
    output_dict['output_stream'] = output_stream
    output_dict['true_label_stream'] = true_label_stream
    output_dict['queue_evol'] = queue_evol
    output_dict['error_evol'] = error_evol
    output_dict['est_evol'] = est_evol
    output_dict['rem_samples'] = rem_samples
    #-------------------------------------------
    return output_dict


def displayResults(output_dict):
    tau = output_dict['tau']
    arrival_arr = output_dict['arrival_arr']
    time_slots = output_dict['time_slots']
    output_stream = output_dict['output_stream'] 
    true_label_stream = output_dict['true_label_stream']
    queue_evol = output_dict['queue_evol']
    error_evol = output_dict['error_evol']
    est_evol = output_dict['est_evol'] 
    rem_samples = output_dict['rem_samples'] 
    #-------------------------------------------------------
    print 'Remaining Samples:', rem_samples
    print 'Len O/p Stream:', len(output_stream)
    error_frac = computeError(output_stream, true_label_stream)
    print '-------\nFinal Output\n--------'
    print 'All samples cleared after no. iterations:',tau
    print 'Accuracy:%.3f'%(1- error_frac)
    print 'Time average of total Queue Length:%.2f'%np.mean(queue_evol)

    if tau <= (time_slots):
        arrival_arr_pad = arrival_arr[:tau]
    else:
        arrival_arr_pad = np.array(list(arrival_arr) + list(np.zeros(tau - time_slots)))

    plt.plot(range(tau), arrival_arr_pad, 'b-', range(tau), queue_evol, 'r--')
    plt.legend(['Arrival', 'Total Queue Length'])
    plt.title('Arrival and Evolution of Queues')
    plt.xlabel('timeslot')
    plt.show()



    plt.plot(range(tau), [1-e for e in error_evol], 'b-')
    plt.title('Evolution of Classification Accuracy')
    plt.xlabel('timeslot')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(range(tau), [e[0] for e in est_evol], 'b-', range(tau),[np.mean(e[1]) for e in est_evol] , 'r--')
    plt.legend(['p_true', 'conf mat'])
    plt.title('Evolution of Estimation Error')
    plt.xlabel('timeslot')
    plt.show()

    
if __name__ =='__main__':
    # Compute the results 
    output_dict = simulator(num_class = 3, num_classifier = 5,  # Ensemble and Data
                  load_ENS = True, fit_ENS = False, 
                  load_params = False, save_params = False, # Ensemble
                  k_max_size = 3, # allowed subset 
                  arr_rate = 1.6, thres = 0.6, V = 0,# arrival and departure thres
                  updateON = True, truePrior = True, # priors
                  time_slots = 200)# simulation length
    # Display the results    
    displayResults(output_dict)
