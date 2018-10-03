
# coding: utf-8

# In[1]:


import numpy as np
from itertools import product
import cvxpy as cvx
import traceback
import itertools
from matplotlib import pyplot as plt
from scipy.stats import entropy
import sys
import random
from collections import Counter


# In[2]:


from Ensembles.Ensemble import Ensemble
from Ensembles.Cifar10.Accessor.readCifar10 import *


# In[3]:

from spectralEMNew import spectralEM
from conflictGraphNew import conflictGraph
from fakeENS import fakeENS


# In[4]:


def kbits(n, k):
    '''Generate power set for set n, limited
       to sets of length k or less'''
    result = []
    for bits in itertools.combinations(range(n), k):
        s = np.zeros(n)
        for bit in bits:
            s[bit] = 1
        result.append(s)
    return result


# In[5]:


def str_arr(arr, mode = 'float'):
    if mode == 'int':
        return " ".join("%2d"%x for x in arr)
    else:
        return " ".join("%.3f"%x for x in arr)
def str_mat(mat):
    return "\n".join(str_arr(arr) for arr in mat)


# In[6]:


def averageAcc(p_true, C):
    '''Weight accuracy for each class by class frequency'''
    return np.sum(C[i,i]*p for i, p in enumerate(p_true))


# In[7]:


def computeError(output_stream, true_label_stream):
    # output_stream: list of sample_out = (id, label_final, new_label, label_conf)
    if len(output_stream) == 0:
        return 0
    est_label = [o[1] for o in output_stream]
    error = [(np.abs(x-y)>0.5) for x,y in zip(est_label, true_label_stream)]
    error_frac = (np.sum(error)/float(len(error)))
    return error_frac



# In[8]:


def ConfError(C_real, num_data, p_true, num_classifier,
                        C_est, ph_est_avg, specFlag):
    # C_real = Array of real confusion matrices
    # num_data = number of data points in simulation
    # p_true = gnd truth class frequency
    # num_classifier = nuimber of classifiers
    # C_est = Array of estimated confusion matrices
    # ph_est_avg = Estiamtion of p_true
    # returns: errp: error  in p_true, err: error in conf matrices
    errp = np.linalg.norm(p_true - ph_est_avg, 1) #L1-Norm
    err = [np.linalg.norm(C_real[k]- C_est[k], 1) for k in range(num_classifier)]
    print '---------------------------'
    print '*****Param Update*****'
    print '---------------------------'
    print 'Spectral Estimation?%r'%specFlag
    print 'Num samples:%d'%num_data
    print 'True ph:%s'%str_arr(p_true)
    print 'Est ph:%s'%str_arr(ph_est_avg)
    print 'L1 error in class probability:%.3f'%errp
    print 'Sum of L1 error in conf matrices:%.3f'%np.sum(err)
    print '---------------------------'
    return errp, err


# In[9]:
def findTrueLabel(id, stream_sample, stream_true_labels):
    ind_s = np.where(stream_sample == id)
    if stream_sample[ind_s] != id:
        print "Error in finding sample location"
        return -1
    else:
        return stream_true_labels[ind_s]+1


def simulator(data_set = 'Cifar10', ens_num = 0,
              num_class = 3, num_classifier = 5,  # Ensemble and Data
              load_ENS = False, fit_ENS = False,
              load_params = True, save_params = False,# Ensemble
              Maxiter= 25, num_init = 10,
              k_max = 3, # allowed subset
              arr_rate = 0.8, thres = 0.75, V = 1, # arrival and departure thres
              updateON = True, truePrior = False, # priors
              expert_frac = 0.1, # false prior: diag = expert_frac+(1-expertfrac)/num_class;
                                 #              non-diag = (1-expertfrac)/num_class
              time_slots = 1000 # simulation length
             ):
    '''
    The function that simulates the system with the input parameters

    '''
    #---------------------------------
    #       ENS and Data
    #---------------------------------
    if load_ENS:
        # Create ensemble of classifiers along with meta-data
        # both for ensemble and for specifying each classifier
        ENS=Ensemble(data_set, ens_num = ens_num)
        ENS.get_train_data(); ENS.assign_members_train_data()
        ENS.get_test_data(); ENS.assign_members_test_data()

        # Instantiate ensemble's classifiers
        ENS.create_classifiers();

        try:
            # fit the ENS again
            if fit_ENS:
                ENS.fit_classifiers()
                ENS.save_classifiers()
            ENS.load_classifiers()
        except:
            print 'Error in loading classifiers'
            ENS.fit_classifiers()
            ENS.save_classifiers()
            ENS.load_classifiers()

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
        # Loading FakeENS and data
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
        ENS = fakeENS(num_classifier, num_class, confMat = conf_mat, excess_acc = [0.85, 0.85, 0.85])
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
    # Prior distribution - uniform probability - unip
    unip = np.ones(num_class)/float(num_class);

    if truePrior:
        # True prior
        params['Confmat'] = trueConfMat
        params['p_true'] = p_true
    else:
        # expert prior with uniform label porbabilities
        # the prior of conf matrices - larger expert_frac => larger diagonal
        expert_frac = 0.1
        expertConfMat = {j: (expert_frac)*np.eye(num_class)+
                         (1-expert_frac)*np.tile(unip, [num_class, 1])
                          for j in range(num_classifier)}
        params['Confmat'] = expertConfMat
        params['p_true'] = unip
    #---------display: current parameter error--------------
    est_err = ConfError(trueConfMat, 0, p_true, num_classifier,
                         params['Confmat'], params['p_true'], False)


    true_avg_acc = [averageAcc(p_true, trueConfMat[i]) for i in range(num_classifier)]
    init_avg_acc = [averageAcc(p_true, params['Confmat'][i]) for i in range(num_classifier)]
    print 'True Avg accuracy: %s'%str_arr(true_avg_acc)
    print 'Initial Avg accuracy: %s'%str_arr([averageAcc(p_true, params['Confmat'][i])
                                              for i in range(num_classifier)])
    #----------------------------------------------------------
    #                Allowed subsets
    #----------------------------------------------------------
    allowed_subset = []
    #               Matching problem/ Generalized assignment problem
    allowed_subset += kbits(int(num_classifier), int(k_max))
    #           Create the conflict graph

    disp = 100    # disp parameter for conflict Graph
    #display: allowed subsets
    print 'Allowed all subsets of size:',k_max
    #----------------------------------------------------------
    #                Conflict Graph
    #----------------------------------------------------------
    G = conflictGraph(allowed_subset = allowed_subset, M = num_classifier,
                      C = num_class, Ens= ENS, params = params,
                      V = V, thres = thres, disp = disp)
    #----------------------------------------------------------
    #         Spectral Estimator
    #----------------------------------------------------------
    S = spectralEM(num_classifier, num_class, maxiter = Maxiter, num_init = num_init, disp =False)
    #----------------------------------------------------------
    #          Metrics for Spectral Estimation
    #----------------------------------------------------------
    group = S.group
    groupConfMat = {i: np.mean([trueConfMat[j]
                                for j in range(num_classifier) if group[i][j]], axis = 0)
                    for i in range(3)}

    kappa = np.min([np.min([[groupConfMat[j][l,l] - groupConfMat[j][l,c]
                             for c in range(num_class) if c !=l]
                          for l in range(num_class)]) for j in range(3)])


    barD = np.min([[np.mean([entropy(trueConfMat[i][l,:], trueConfMat[j][l,:])
                             for i in range(num_classifier)])
                    for c in range(num_class) if c !=l]
                   for l in range(num_class)])

    #Display: metrics for spectral estimation
    print 'Metrics in Spectral Learning| kappa:%.3f, barD:%.3f'%(kappa, barD)
    print
    #---------------------------------------------------
    #              Reset Internal States
    #---------------------------------------------------
    G.reset() # resets the internal queues
    S.reset() # resets the internal parameters
    #       Exploration and estimation parameters
    update_num = 5  # lenght of explore_data when S is updated
    updateSpectral = True # no spectral updates if False (only EM)
    updateEM = True # no EM updates  if False (only Spectral)
    Spec2EM = 10
    # make the following two 0 to stop exploration
    explore_prob_const = 0.1 # probability with which exploration happens
    init_explore = 750 # Number of initial time slots when explore happens
    #---------------------------------------------------
    # Initialization
    output_stream= []
    true_label_stream = []
    queue_evol = []
    error_evol = []
    est_evol = []
    spec_evol = []

    samp_count = 0
    tau = 0
    explore_data = []
    count_EM_update = 0

    # display period
    disp_period = 25
    display = 2 # display flag

    print 'Arrival Rate:', arr_rate, 'Accuracy Threshold:', G.thres
    print 'Total samples:', num_tot_samp, 'Total time slots:', time_slots
    print 'Update the parameters:', updateON
    print 'Update using spectral method:', (updateON and updateSpectral)
    #--------------------------------------------------------
    #    Update the Spectral Estimator with Intial Data
    #--------------------------------------------------------
    # Select Init_explore Number of samples from the stream_sample
    if updateON:
        init_explore = min(num_tot_samp, 500)
        print '---------------------'
        print ' Initial Exploration '
        print '---------------------'
        explore_ids = np.random.choice(range(num_tot_samp), size = init_explore)
        G.newArrival(stream_sample[explore_ids]+1)
        count_init_explore = 0
        init_explore_data = []
        while (count_init_explore < (init_explore-1)):
            _, _, new_labels_out = G.schedule(explore = True)
            init_explore_data += [new_labels_out-1]
            count_init_explore +=len(init_explore_data)
            S.update(init_explore_data)
            print 'Number of exploration samples Added:%d'%count_init_explore
        G.reset() # Reset G
    #--------------------------------------
    #            Iterations
    #--------------------------------------
    print '--------------------'
    print '  Iterations Begin  '
    print '--------------------'
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
        explore_prob = explore_prob_const
        #explore_prob = explore_prob_const*np.log(i)/i # epsilon_t Bandit
        explore_flag = (np.random.rand() < explore_prob) and updateON # explore only if updateON
        specFlag = False
        EMFlag = False
        #----------------------------------------------------------
        #                         Schedule
        #----------------------------------------------------------
        try:
            sample_out, schedule, new_labels_out = G.schedule(explore = explore_flag)
            # sample_out: (id, label_final, new_label, label_conf)
        except Exception as e:
            print '***Error in Scheduling***Exiting***'
            traceback.print_exc(); break

        #----------------------------------------------------------
        #                  Update Parameters
        #----------------------------------------------------------
        if updateON: # update on each output
            if explore_flag: # update only on exploration instances
                explore_data += [new_labels_out-1]
            #print 'explore_data:', explore_data

        if (len(explore_data) >= update_num) and updateON:

            #         update the spectralEM object

            S.update(explore_data);

            #          recallibrate params using Spectral

            if updateSpectral and (count_EM_update >= Spec2EM):
                S.updateParamsSpectral()
                count_EM_update = 0
                specFlag = True

                #   update params using EM
            if updateEM:
                S.updateParamsEM();
                count_EM_update +=1
                EMFlag = True

            #           update G params

            new_params = {'Confmat': S.conf_mat, 'p_true': S.ph_est_avg}
            G.updateParams(new_params)


            est_err = ConfError(trueConfMat, S.num_data, p_true,
                           num_classifier, S.conf_mat, S.ph_est_avg, specFlag)

            #           empty explore_data

            explore_data = []
        #----------------------------------------------------------
        #                    Results
        #----------------------------------------------------------
        true_label_out = []
        #print 'sampleout:',sample_out
        for s in sample_out:
            true_label_out += [findTrueLabel(int(s[0]-1), stream_sample, stream_true_labels)]


        rem_samples = G.totalCount()
        output_stream += sample_out
        tot_queue = G.totalCount()
        tot_actQ = G.totalQueues()
        true_label_stream += true_label_out
        classification_error = computeError(output_stream, true_label_stream)


        queue_evol.append(tot_queue)
        error_evol.append(classification_error)
        est_evol.append(est_err)
        spec_evol.append(specFlag)
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
            print 'Iteration:', tau, 'Exploration?', explore_flag
            print 'Schedule:', schedule
            #print 'New samples for learning:%s'%str_arr(new_labels_out, 'int')
            print 'New samples for learning:',new_labels_out
            print 'Output'
            for i_s, s in enumerate(sample_out):
                # s: sample_id, label_final, label_arr, label_conf
                print " ".join(['Sample Id:%4d'%s[0],
                                'Labels=> Real:%2d'%(true_label_out[i_s]),
                                'Final:%2d'%s[1], 'Conf:%.3f'%s[3],
                                'All: %s'%str_arr(s[2], 'int')])

            print '# Active queues:', tot_actQ, 'Sum queue length:', tot_queue
            print

        #  increase the time count

        tau += 1
    #----------------------outputs-----------
    output_dict = {}
    output_dict['tau'] = tau
    output_dict['time_slots'] = time_slots
    output_dict['arrival_arr'] = arrival_arr
    output_dict['output_stream'] = output_stream # sample_out :
    output_dict['true_label_stream'] = true_label_stream
    output_dict['queue_evol'] = queue_evol
    output_dict['error_evol'] = error_evol
    output_dict['est_evol'] = est_evol
    output_dict['spec_evol'] = spec_evol
    output_dict['rem_samples'] = rem_samples
    output_dict['trajectories'] = G.trajectories
    output_dict['classifierHist'] = G.classifierHist
    #-------------------------------------------
    return output_dict


# In[10]:


def displayResults(output_dict):
    tau = output_dict['tau']
    arrival_arr = output_dict['arrival_arr']
    time_slots = output_dict['time_slots']
    output_stream = output_dict['output_stream']
    true_label_stream = output_dict['true_label_stream']
    queue_evol = output_dict['queue_evol']
    error_evol = output_dict['error_evol']
    est_evol = output_dict['est_evol']
    spec_evol = output_dict['spec_evol']
    rem_samples = output_dict['rem_samples']
    #-------------------------------------------------------
    print 'Remaining Samples:', rem_samples
    print 'Len O/p Stream:', len(output_stream)
    error_frac = computeError(output_stream, true_label_stream)
    print '-------\nFinal Output\n--------'
    print 'All samples cleared after no. iterations:',tau
    print 'Accuracy:%.3f'%(1- error_frac)
    print 'Time average of total Queue Length:%.2f'%np.mean(queue_evol)

    print '---------\n Sample Trajectories\n--------'
    trajs = random.sample(output_dict['trajectories'].items(), 10) # trajectories
    num_used_classifier = []
    for k,v in trajs:
        print 'id:%d'%k
        for i,n in enumerate(v):
            print 'Step:%d Labels:%s  Posterior:%s '%(i, str_arr(n[0], 'int'), str_arr(n[1]))
            if len(n) == 3:
                print 'Final Label:%d'%n[2]
                num_used_classifier += [np.sum(n[0]>0)]
        print

    print 'Average number of classifiers used per sample:',np.mean(num_used_classifier)

    print '---------\n Classifier Histogram\n---------'
    for k,hist in output_dict['classifierHist'].items():
        print 'Classifier:%d'%k
        print 'Histogram'
        for l, L in sorted(hist.items()):
            if len(L) > 10:
                print 'label:%s, #samples:%d'%(l,len(L)),'|'
        print

    if tau <= (time_slots):
        arrival_arr_pad = arrival_arr[:tau]
    else:
        arrival_arr_pad = np.array(list(arrival_arr) + list(np.zeros(tau - time_slots)))
    # Plot Queue Lengths
    plt.plot(range(tau), arrival_arr_pad, 'b-', range(tau), queue_evol, 'r--')
    plt.legend(['Arrival', 'Total Queue Length'])
    plt.title('Arrival and Evolution of Queues')
    plt.xlabel('timeslot')
    plt.show()

    # Plot Accuracy
    plt.plot(range(tau), [1-e for e in error_evol], 'b-')
    plt.title('Evolution of Classification Accuracy')
    plt.xlabel('timeslot')
    plt.ylabel('Accuracy')
    plt.show()

    # Plot Estimation Error
    plt.plot(range(tau), [e[0] for e in est_evol],
             'b-', range(tau),[np.mean(e[1]) for e in est_evol] , 'r--')
    plt.legend(['p_true', 'conf mat'])
    # instances of spectral Estimation
    for i in range(tau):
        if spec_evol[i]:
            plt.axvline(x=i)

    plt.title('Evolution of Estimation Error')
    plt.xlabel('timeslot')
    plt.show()


# In[15]:


if __name__ =='__main__':
    # Compute the results
    # arg1 = arr_rate, arg2 = time_slots, arg3 = num_class,
    # arg4 = num_classifier, arg5 = threshold
    #--------------------------------
    print sys.argv
    arg = [0.9, 200, 2, 5, 0.75, False]
    floats = [1,5]
    for i in range(1, len(sys.argv)):
        if i in floats:
            arg[i-1] = float(sys.argv[i])
        else:
            arg[i-1] = int(sys.argv[i])
    #--------------------------------
    output_dict = simulator(
                      arr_rate = arg[0], # arrival
                      time_slots = arg[1],# simulation length
                      num_class = arg[2], # Number of classes
                      num_classifier = arg[3],  # NUmber of Classfiers
                      thres = arg[4],# departure accuracy thres
                      updateON = arg[5], # updates the parameters iff True
                      #---------------Prior on params---------------
                      truePrior = False, # uses True prior iff True
                      expert_frac = 0.1, # if false prior: diag = expert_frac+(1-expertfrac)/num_class;
                                         #              non-diag = (1-expertfrac)/num_class
                      Maxiter = 75, num_init = 100, #Tensor decomposition parameters
                      k_max = 1, # maximum size of the allowed subsets
                      #--------------The Ensemble Type--------------------
                      load_ENS = False, # NN clasifiers used iff load_ENS = True, else fake_ENS is used
                      fit_ENS = False, # NN training is done for NN classifiers iff fit_ENS = True,
                      ens_num=0, # NN Ensemble Number
                      load_params = False, # params are loaded from a file if load_params = True
                      save_params = False, # params are saved in a file if save_params = True
                      #----------------------------------------------
                  )
    # Display the results
    displayResults(output_dict)
