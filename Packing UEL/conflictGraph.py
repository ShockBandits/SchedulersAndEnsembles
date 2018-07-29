
# coding: utf-8

# In[1]:


import numpy as np
from itertools import product
import cvxpy as cvx
import traceback
import itertools
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


# In[2]:


class labelSubset:
    '''
    Maintains the labelqueues where labels are obtained from the active subset, self.subset. 
    Active subset is the binary encoding of the machines used for obtaining labels.
    Each labelqueue contains all the sample with a specific label.
    '''
    
    def __init__(self, subset, estimator, confidence, disp = 100):
        # subset:- binary encoding of the active machines
        # queues:-  cache of labelqueues corresponding to the labels
        self.subset = subset
        self.queues = {} # Dict: key: label, value: [count, list of the element ids]
        self.est = estimator
        self.conf = confidence
        self.disp = disp
        
    def printLS(self):
        print 'subset:%s'%str(self.subset)
        if len(self.queues.items()) > 0:
            print 'Queues(label:count): %s'%",".join(k+':'+str(v[0])
                                                      for k,v in self.queues.items())
        
    def queueEntry(self, label, entry):
        # label: (not a string) np array of labels
        # entry: the sample id (begins from 1)
        # increase queue length
        
        # check for the label being in the active subset
        chk1 = (np.dot(label, [not i for i in self.subset]) == 0)
        # check if all lables are nonzero inside the active subset
        chk2 = (np.dot((np.array(label) == 0), [i for i in self.subset]) == 0)
            
        if chk1 and chk2:
            if str(label) in self.queues.keys():
                # already present
                self.queues[str(label)][0] += 1
                self.queues[str(label)][1] += [entry]
            else:
                # new entry
                self.queues[str(label)] = [1, [entry], label]
        else:
            print 'Invalid entry: subset:%s, label:%s, entry:%s'%(str(self.subset), 
                                                                  str(label), str(entry))
            if not chk1: 
                print 'Error! Labels are marked outside active set'
            if not chk2:
                print 'Error! All labels are not marked in active set'
    
    def queueExit(self, label):
        # label is a string
        # return a random element from the queue label
        if label in self.queues.keys():
            # select a random element from the list
            #exit = np.random.choice(self.queues[label][1]) # random choice
            exit = self.queues[label][1][0] # oldest first
            exit_label = self.queues[label][2]
            self.queues[label][1].remove(exit)
            self.queues[label][0] -= 1
            # remove a queue if empty
            if self.queues[label][0] == 0:
                del self.queues[label]
            return exit, exit_label
        else:
            # list is empty
            if self.disp < 10:
                print '-------------------'
                print 'Warning!!: Returning None in queueExit'
                print 'subset:', self.subset
                print 'Queue:', self.queues
                print 'Label:', label
                print '-------------------'
            return None, None
    
    def totalCount(self):
        # returns total number of elements in the queue
        return np.sum([v[0] for v in self.queues.values()])
    
    def queueMax(self):
        # returns the max label and its queuelength
        keys = self.queues.keys()
        # O(#labelqueues) ops; can be expensive
        # Need to maintain a maxheap for O(log(#labelqueues)) ops
        if len(keys) > 0:
            id_max = keys[np.argmax([self.queues[k][0] for k in keys])]
            value_max = self.queues[id_max][0]
            label_max = self.queues[id_max][2]
            return label_max, value_max
        else:
            return None, 0 
    
    def queueWA(self, label_in):
        # self.queues: keys: str(label), values: [count,[array of ids], label]
        # given the label_in, sum_label P(label_in->label)*len(label)
        return np.sum([self.est(label_in, self.queues[label][2])*self.queues[label][0] 
                       for label in self.queues.keys()])
    
    def incntvWA(self,label_in):
        # self.queues: keys: str(label), values: [count,[array of ids], label]
        # given the label_in, sum_label P(label_in->label)*conf(label)
        # conf(label) = (max_c P(true = c| label) - thres)*(label == terminal)
        # confidence can be negative: (This deters entering all active state if possible)
        return np.sum([self.est(label_in, self.queues[label][2])*self.conf(self.queues[label][2])
                       for label in self.queues.keys()])


# In[5]:


class conflictGraph:
    ### A subset of machines assigned to a specific sample
    ### It creates a link between two labelSubset nodes
    ### It is a node in the conflict graph
    def __init__(self, allowed_subset, M, C, 
                 Ens, params, V = 1, thres = 0.9, disp = 100):
        # allowed_subset: list of allowed subset
        # M: number of classifier number
        # minM: minimum number of classifiers for final classification
        # C: number of classes
        # Ens: The classifier ensemble
        # params: the estimated hidden parameters 
        self.params = params # the estimated hidden parameters 
        self.allowed_subset = allowed_subset
        self.classifier_num = M
        self.num_classes = C
        self.classifierEnsemble = Ens
        self.thres = thres
        self.V = V
        #-------parallel
        self.num_cores = min(20,multiprocessing.cpu_count())
        #-------------------
        self.disp = disp
        # high value of self.disp suppresses more print statement
        # self.disp = 0 shows all print statement
        # self.disp = 100 hides all print statement
        # Ens.classify(schedule)
        #  i\p. schedule is a list of sample id
        #  o\p. list of labels from the classifiers for the respective sample ids
        self.__createconflictGraph()
        #-----display-----
        if self.disp <= 1: 
            print 'labelSubsetList:', self.labelSubsetList;
            print 'Connex dict/ Nodes:', self.connexDict;
            print 'Conflict Edges/ Edges:', self.edges;
            print 'Neighbors:', self.neighbor;
        #------------------
        # Create the label subsets
        self.labelSubsets = {str(s): labelSubset(s, self.__estimator, self.__confidence) 
                             for s in self.labelSubsetList}
        
    
    def reset(self):
        # Delete the labelSubsets
        for k in self.labelSubsets.keys():
            del self.labelSubsets[k]
        # ReCreate the label subsets
        self.labelSubsets = {str(s): labelSubset(s, self.__estimator, self.__confidence) 
                             for s in self.labelSubsetList}
    
    def totalCount(self):
        return np.sum([lS.totalCount() for lS in self.labelSubsets.values()])
            
        
    def print_labelSubsets(self):
        for lS in self.labelSubsets.values():
            lS.printLS()
            
    def updateParams(self, new_params):
        self.params = new_params # update hidden params
        
    def __createconflictGraph(self):
        labelSubsetList = [np.zeros(self.classifier_num).astype(int)]
        connexDict = {str(s):[] for s in self.allowed_subset}
        curr = 0
        while curr < len(labelSubsetList):
            label_curr = labelSubsetList[curr]
            #------display-------
            if self.disp <= 0: print 'label curr:', label_curr;
            #-------------
            for s in self.allowed_subset:
                label_new = np.array(np.minimum(1, label_curr+s))
                min_dist = np.min([np.linalg.norm(label_new - l) 
                                   for l in labelSubsetList])
                pair_dist = np.linalg.norm(label_new - label_curr)
                #------display-------
                if self.disp <= 0: 
                    print 's:', s, ' min_dist:', min_dist, ' label_new:', label_new
                #-------------
                if pair_dist > 0: connexDict[str(s)]+= [(label_curr, label_new)];
                if min_dist > 0: labelSubsetList += [label_new];
            curr +=1
        # list of labelSubsets
        self.labelSubsetList = labelSubsetList
        # list of connection dictionaries (nodes)
        self.connexDict = connexDict # Also the nodes of the conflict graph
        # list of connection edges (edges)
        self.edges = []
        self.neighbor = {str(s): [] for s in self.allowed_subset}
        for x, y in product(self.allowed_subset, repeat =2):
            if (np.sum(np.multiply(x,y))>0) and (np.linalg.norm(x-y)>0):
                self.edges += [(str(x),str(y))]
                self.neighbor[str(x)] += [str(y)]
                
    #-------------------------------------------------------------
    ## The following are the inputs from the UEL module
    ## And some meaning less estimator
        
    def __labelCondclass(self, c, label):
        # label probability given true class
        # C_i : confusion matrix for classifier i
        # P(label|c) = prod_(i: label[i]>0) C_i(c, label[i]) 
        conf_mat = self.params['Confmat']
        arr = []
        for i, l in enumerate(label):
            if l > 0:
                val = conf_mat[i][int(c), int(l-1)]
                if self.disp <= 1: print 'i,l,c, val:', i, l, c, val;
                arr += [val]
        return np.prod(arr)

    def __labelProb(self, label):
        # label probability
        # P(label) = sum_c P(label|c)P(c)
        p_true = self.params['p_true']
        conf_mat = self.params['Confmat']
        res = 0
        for c in range(self.num_classes):
            res += self.__labelCondclass(c, label)*p_true[c]
        return res
        # One line expression
        # return np.sum(np.prod([conf_mat[i][c, int(l-1)] 
        # for i, l in enumerate(label) if l > 0])*p_true[c] 
        # for c in range(self.num_classes))
       
    def __classCondlabel(self, c, label):
        # class probability given label
        # P(c|label) = P(label|c)P(c)/P(label)
        p_true = self.params['p_true']
        conf_mat = self.params['Confmat']
        return (self.__labelCondclass(c, label)*p_true[int(c)])/self.__labelProb(label)

    def __estimator(self, label_in, label_out):
        '''
        Given label_in, label_out outputs: P(label_out|label_in) 
        = \sum_{c} P(label_out|c, label_in) P(c|label_in)
        = 1(label_out 'match' label_in)\sum_{c} P(label_out|c, label_in)
        *P(label_in|c)P(c)/ (\sum_{c'} P(label_in|c')P(c'))
        = 1(label_out 'match' label_in)(\sum_{c} P(label_out|c)P(c))/
                                                (\sum_{c'} P(label_in|c')P(c'))
        = 1(label_out 'match' label_in)P(label_out)/P(label_in)
        '''
        mismatch = np.linalg.norm(label_in*label_out.astype(bool)
                                  -label_out*label_in.astype(bool))
        
        if mismatch > 0: 
            return 0
        else: 
            new_label = label_out - label_out*label_in.astype(bool)
            p_in = self.__labelProb(label_in) # compute the probability of label_in
            p_out = self.__labelProb(label_out) # compute the probability of label_out
            #-------------
            if self.disp <= 3:
                print 'label_in:', label_in, 'p_in:', p_in
                print 'new_label:', new_label, 'p_new:', p_out/float(p_in)
            #--------------
            return p_out/float(p_in)
    
    def __confidence(self, label):
        # given label_in, label_out outputs: 
        # (max_c P(true = c| label) - thres)*(label == terminal) or
        # (max_c P(true = c| label))*(label == terminal)
        _, best_val = self.__checkTerminationBayes(label)
        if best_val is None: 
            return 0
        else:
            #return max(0,best_val - self.thres)
            return best_val
    
    
    def __checkTerminationBayes(self, label):
        prob_c = [self.__classCondlabel(c, label) for c in range(self.num_classes)]
        best_c = np.argmax(prob_c)
        best_val = prob_c[best_c]
        # best confidence above thres or 
        # samples from all classifiers 
        if (best_val > self.thres) or (np.all(label>0)):
            return best_c+1, best_val
        else:
            return None, None
    #----------------------------------------------------------------------------
    def __innerCW(self, l):
        # value_left:max value for the source lS
        label_in, value_left = self.labelSubsets[str(l[0])].queueMax()
        #---------------------
        if label_in is not None:
            # value_right:average value for the destination lS
            incntv = self.labelSubsets[str(l[1])].incntvWA(label_in)
            value_right = self.labelSubsets[str(l[1])].queueWA(label_in)
            value_curr = (value_left - value_right)
            return (value_curr, incntv, label_in, l)
        else:
            return (-1000, None, None, None)
    #----------------------------------------------------------------------------
    def __compute_weights(self):
        # computes weights for each node in the conflict graph
        # each node has a list of (source, dest) pairs of labelSubsets
        # weight = max(source.queueMax() - dest.queueWA(label_in) , over all pairs)
        maxweightDict = {}
        for k, L in self.connexDict.items(): # O(|S|)
            # Subset k is one among the allowed subsets
            # L is the labelSubset pairs connected by Subset k (# O(2^m)) 
            #res_arr = Parallel(n_jobs = self.num_cores)(delayed(self.__innerCW)(l) for l in L)
            res_arr = [self.__innerCW(l) for l in L]
            #------choosing the max
            max_ind = np.argmax([r[0] for r in res_arr]) 
            max_res = res_arr[max_ind]
            if (max_res[0])>0:
                value_max = max_res[0]; incntv_max = max_res[1];
                label_max = max_res[2]; l_max = max_res[3];
                maxweightDict[k] = [value_max, l_max, str(label_max), incntv_max]
            #else: for all l in L the source was empty
            #   do nothing
        #-------------------------
        #------display-------
        if self.disp <=2: print 'maxweightDict:\n', maxweightDict;
        #--------------------
        return maxweightDict
                
    def __compute_schedule(self, maxweightDict):
        # produces a schedule for the classifiers
        # maxweightDict keys: each node, values: [value_max, l_max, label_max, incntv_max]
        # value_max = weight of node, 
        # l_max = corresponding pair of labelSubsets, l_max[0]: source, l_max[1]: dest
        # label_max = corresponding label in l_max[0] (source labelsubset)
        #-------------------------------
        # Solves a maximum independent set problem on the conflict graph
        # Uses a MILP formulation for smaller instances
        # Uses an LP relaxation and a greedy rounding for larger instances[1]
        # 1. Kako et al., Approximation Algorithms for the Weighted Independent Set Problem
        # Next for the active nodes, we select the sample under the given label_max
        #---------------------------------
        nodes = maxweightDict.keys()
        num_nodes = len(nodes)
        num_edges = len(self.edges)
        #------display-------
        if self.disp <= 1:
            print '\nMaxweight:', maxweightDict
        #---------------------
        if num_nodes == 0:
            if self.totalCount() > 0:
                print '!!!Error!!!'
                print 'Nothing to schedule but Queue length:',self.totalCount()
                print 'The current state:'
                self.print_labelSubsets()
                print
            return np.zeros(self.classifier_num), {}
        #---------------------------------
        ## Solving the Max Weighted Independent Set problem.
        x = cvx.Variable((num_nodes,1), boolean = True) #MILP
        #-----------------------------------------
        # constraints
        #-----------------------------------------
        # 1. Independent Set constraints
        #-----constraint matrix----
        A = np.zeros((num_edges, num_nodes))
        for i, edge in enumerate(self.edges):
            if edge[0] in nodes: 
                j1 = nodes.index(edge[0]); A[i,j1] = 1; 
            if edge[1] in nodes: 
                j2 = nodes.index(edge[1]); A[i,j2] = 1;
        #-----------------------------------------
        # 2. Non repetition of samples constraint
        # dict_maxlS: key-label queue, value[0]- #entries in queue, 
        #           value[1]-list of nodes connected to label queue
        dict_maxlS = {}
        for node, val in maxweightDict.items():
            maxlS = val[2]
            w_maxlS = val[0]
            try:
                dict_maxlS[maxlS][1] += [node] 
            except:
                dict_maxlS[maxlS] = [w_maxlS, [node]]

        num_maxlS = len(dict_maxlS.keys())
        B = np.zeros((num_edges, num_nodes))
        b = np.zeros((num_edges, 1))
        
        list_maxlS = dict_maxlS.keys()
        
        for i, maxlS in enumerate(list_maxlS):
            j1 = [nodes.index(n) for n in dict_maxlS[maxlS][1]]; 
            B[i,j1] = 1; b[i] = dict_maxlS[maxlS][0]
        #-----------------------------------------
        constraints = [B*x <= b, A*x <= np.ones((num_edges, 1)), 0 <= x, x <= 1]
        #-----------------------------------------
        # objective
        #-----------------------------------------
        bp = np.array([maxweightDict[n][0] for n in nodes])
        incntv = np.array([maxweightDict[n][3] for n in nodes])
        w = bp+self.V*incntv
        if self.disp <= 1:
            print 'weight data (n,bp, incntv, w):',[(n,bp[i], incntv[i], w[i]) 
                                     for i,n in enumerate(nodes)]
        # Form objective
        obj = cvx.Maximize(w.T*x)
        #-----------------------------------------
        # Form and solve problem
        #-----------------------------------------
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver = cvx.ECOS_BB)  # Returns the optimal value
        #-----------------------------------------
        try:
            node_active = [nodes[i] for i, t in enumerate(x.value) if t > 0.5]
            if (self.totalCount() > 0) and (len(node_active) == 0):
                print '!!!Error!!!'
                print 'Nothing to schedule but Queue length:',self.totalCount()
                print("status:", prob.status)
                print('Maxweight:keys: each node, values: [value_max, l_max, label_max, incntv_max]')
                print maxweightDict
                print ("dict_maxlS: key:labelqueue, value:[# labelqueue, list connected nodes]")
                print dict_maxlS
                print 'The current state:'
                self.print_labelSubsets()
                print 
        except Exception as e:
            #------------
            print '!!!Error!!!!!'
            print e
            print("status:", prob.status)
            print("optimal value", prob.value)
            print("optimal var", x.value)
            print('Maxweight:keys: each node, values: [value_max, l_max, label_max, incntv_max]')
            print maxweightDict
            print ("dict_maxlS: key:labelqueue, value:[# labelqueue, list connected nodes]")
            print dict_maxlS
            print 
            #------------
            return np.zeros(self.classifier_num), {}
        #------display------------------------------
        if self.disp <= 1:
            print("status:", prob.status)
            print("optimal value", prob.value)
            print("optimal var", x.value)
        #--------------------------------------------
        # With the MWIS problem solved we need to form the schedule
        schedule = np.zeros(self.classifier_num)
        samplesOnTheFly = {}
        for node in node_active:
            schedule_old = np.copy(schedule)
            
            source_lS = maxweightDict[node][1][0] # lS = labelSubset
            dest_lS = maxweightDict[node][1][1]
            lMax = maxweightDict[node][2] #lMax = labelMax
            
            # Get the sample Id from: lMax and source_lS
            sample_id, sample_label = self.labelSubsets[str(source_lS)].queueExit(lMax)
            
            if sample_id is not None:
                # lMax = str(sample_label), sample_lable is np array
                samplesOnTheFly[sample_id] = (sample_label, dest_lS)
                # New Classifiers: The node
                new_classifiers = (dest_lS - source_lS) 
                # Add the new_classifier schedule
                schedule += new_classifiers*sample_id
            # else: Skip
            
            #------display-------
            if self.disp <= 1:
                print('node:',node, 'source:', source_lS, 'dest:', 
                       dest_lS, 'label:', lMax, 'sample_id:',sample_id)
                print 'schedule old:', schedule_old, 'schedule new:', schedule
            #---------------------
            # Why 'sample_id == None' occurs? 
            # More than one nodes may have the same source as the best choice in maxweight
            # In such situations, scheduling one node may make a non-empty queue empty
        
        #------display-------
        if self.disp <= 1:
            print 'MWIS:', node_active
            print 'sch:',schedule
            print 'samples:',samplesOnTheFly
        #--------------------
        return schedule, samplesOnTheFly
            
    def __explore_schedule(self):
        # pick a sample from label Subset label_0 if exists
            curr = 0; samplesOnTheFly  = {};
            schedule = np.zeros(self.classifier_num);
            
            dest_lS = np.ones(self.classifier_num);
            new_classifiers = np.ones(self.classifier_num);
            #----------
            for source_lS in self.labelSubsetList:
                # find the label
                label_in, _ = self.labelSubsets[str(source_lS)].queueMax()
                # find the sample
                sample_id, _ = self.labelSubsets[str(source_lS)].queueExit(str(label_in))
                #print 'explore schedule:', source_lS, label_in, sample_id, sample_label
                #------------
                if sample_id is not None:
                    # lMax = str(sample_label), sample_label is np array
                    samplesOnTheFly[sample_id] = (np.zeros(self.classifier_num), dest_lS)
                    # here we reset existing label as we send it to all classifiers
                    schedule = new_classifiers*sample_id
                    break
            #-------------  
            return schedule, samplesOnTheFly
    
    def schedule(self, explore = False):
        # schedules the samples to classifiers
        # steps: 1) update the weights for each key in connectDict
        #        2) solve a max weighted independent set with computed weights
        #        3) schedule labels to classifiers according 
        #        4) Augment the new labels and move to the particular labelqueues
        # maxweight policy: step 1 and 2
        #------------------------------
        maxweightDict = self.__compute_weights()
        # maxweightDict keys: each node, values: [value_max, l_max, label_max]
        # value_max = weight of node, 
        # l_max = corresponding pair of labelSubsets, l_max[0]: source, l_max[1]: dest
        # label_max = corresponding label in l_max[0] (source labelsubset)
        #------------------------------
        if explore:
            schedule, samplesOnTheFly = self.__explore_schedule()
        else:
            schedule, samplesOnTheFly = self.__compute_schedule(maxweightDict)
        # schedule[i] = sample id scheduled to classifier i (0 = No sample)
        new_labels = self.classifierEnsemble.classify(schedule)
        # new_labels[i] = label coming from the i-th classifier
        # new_labels has to be list of len (num_classifier) 
        #                or np.array of dim (num_classifier,1)
        #------------------------------
        # Update the queues now
        # sample_id >= 1 (0 is reserved for no entry to the classifier)
        sample_out = []
        for sample_id, value in samplesOnTheFly.items():
            old_label, dest_lS = value
            fresh_labels = np.multiply(new_labels, np.array(schedule) == sample_id)
            #-----display-------
            if self.disp <= 1:
                 print('old label:', old_label, 'fresh_label:', fresh_labels) 
            #---------------------
            new_label = old_label + fresh_labels
            # put the sample id with its label in the apt labelSubset queue
            label_final, label_conf = self.__checkTerminationBayes(new_label)
            if label_final is not None:
                #-----display-------
                if self.disp <= 10:
                    print('new_label:', new_label, 'sample_id:%d'%sample_id, 
                    'label_final:%d'%label_final, 'label_conf:%.3f'%label_conf)
                #---------------------
                sample_out +=[(sample_id, label_final, new_label, label_conf)]
            else:
                self.labelSubsets[str(dest_lS)].queueEntry(new_label, sample_id)
        del samplesOnTheFly # nothing on the fly
        return sample_out, schedule, new_labels
        
        
    def newArrival(self, list_new_samples):
        # list_new_samples: list of new samples (id of each sample)
        label_0 = np.zeros(self.classifier_num).astype(int)
        # put the new arrivals in label_0 queue
        for entry in list_new_samples: 
            self.labelSubsets[str(label_0)].queueEntry(label_0, entry) 

