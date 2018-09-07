
# coding: utf-8

# In[1]:


import numpy as np
from itertools import product, combinations
import cvxpy as cvx
import traceback
import itertools
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from collections import deque


# In[2]:


class labelQueue:
    '''
    Maintains the labelqueues where labels are obtained from the active subset, self.subset.
    Active subset is the binary encoding of the machines used for obtaining labels.
    Each labelqueue contains all the sample with a specific label.
    '''

    def __init__(self, label, disp = 100):
        # label:- a specific label: label[i]= label from classifier i if > 0 else not labelled
        # queues:-  cache of labelqueues corresponding to the labels
        self.label = label
        self.queue = deque()
        self.disp = disp

    def printLS(self):
        print 'label:%s'%str(self.label)
        print 'Queue length:%s'%str(len(self.queue))

    def queueEntry(self, entry):
        # entry: the sample id (begins from 1)
        # puts the new sample in the queue

        self.queue.append(entry)


    def queueExit(self):
        # return a random element from the queue label
        if len(self.queue)> 0:
            # select a random element from the list
            #exit = np.random.choice(self.queues[label][1]) # random choice
            exit = self.queue.popleft()
            return exit
        else:
            # list is empty
            if self.disp < 10:
                print '-------------------'
                print 'Warning!!: Returning None in queueExit'
                print 'subset:', self.subset
                print 'Queue:', self.queues
                print 'Label:', label
                print '-------------------'
            return None

    def totalCount(self):
        # returns total number of elements in the queue
        return len(self.queue)




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
        self.trajectories = {}
        self.classifierHist = {i:{} for i in range(M)}
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
        self.labelQueues = {}
        #-----display-----
        if self.disp <= 1:
            print 'labelQueueList:', self.labelQueueList;
            print 'Connex dict/ Nodes:', self.connexDict;
            print 'Conflict Edges/ Edges:', self.edges;
            print 'Neighbors:', self.neighbor;
        #------------------

    def __getlabelSubset(self, label):
        return np.array(np.array(label)>0).astype(int)

    def __listToStr(self, L):
        s  = ''
        for l in L:
            s += str(int(l))
        return s

    def __strToList(self, s):
        L  = []
        for ch in s:
            L.append(int(ch))
        return L
    #---------------------------
    def reset(self):
        # Delete the labelSubsets
        for k in self.labelQueues.keys():
            del self.labelQueues[k]

    def totalCount(self):
        return np.sum([lS.totalCount() for lS in self.labelQueues.values()])

    def totalQueues(self):
        return len(self.labelQueues.keys())


    def print_labelQueues(self):
        for lS in self.labelQueues.values():
            lS.printLS()

    def updateParams(self, new_params):
        self.params = new_params # update hidden params

    def __createconflictGraph(self):
        # list of connection edges (edges)
        self.edges = []
        for x, y in product(self.allowed_subset, repeat =2):
            if (np.sum(np.multiply(x,y))>0) and (np.linalg.norm(x-y)>0):
                self.edges += [(self.__listToStr(x), self.__listToStr(y))]
        # the Adjacency matrix for the conflictGraph
        allowedSS = [self.__listToStr(s) for s in self.allowed_subset]
        num_allowedSS = len(allowedSS)
        num_edges = len(self.edges) # edges:
        A = np.zeros((num_edges, num_allowedSS))
        for i, edge in enumerate(self.edges):
            if edge[0] in allowedSS:
                j1 = allowedSS.index(edge[0]); A[i,j1] = 1;
            if edge[1] in allowedSS:
                j2 = allowedSS.index(edge[1]); A[i,j2] = 1;
        self.A = A
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


    def __checkTerminationBayes(self, label, id):
        prob_c = [self.__classCondlabel(c, label) for c in range(self.num_classes)]
        best_c = np.argmax(prob_c)
        best_val = prob_c[best_c]
        # best confidence above thres or
        # samples from all classifiers
        if (best_val > self.thres) or (np.all(label>0)):
            self.trajectories.setdefault(id, []).append((label, prob_c, best_c+1))
            return best_c+1, best_val
        else:
            self.trajectories.setdefault(id, []).append((label, prob_c))
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
        for k in self.allowed_subset: # O(|S|)
        # Subset k is one among the allowed subsets
            for l in self.labelQueues.keys(): #O(|lQs|)
            # l is one label queue
                label_src = [int(i) for i in l]
                subset_src = self.__getlabelSubset(label_src)
                subset_new = np.array(np.minimum(1, subset_src+k))
                subset_diff = [j for j in range(self.classifier_num) if (subset_new[j]>subset_src[j])]
                #-------------
                # create all the possible labels under the new subset
                if len(subset_diff) > 0:
                    list_labels = [[]]*self.classifier_num
                    for lli in range(self.classifier_num):
                        if lli in subset_src:
                            list_labels[lli].append(label_src[lli])
                        elif lli in subset_diff:
                            list_labels[lli].extend(range(1, self.num_classes+1))
                        else:
                            list_labels[lli].append(0)

                    dest_list =  [list(e) for e in product(*list_labels)]



                    Q_l = self.labelQueues[l].totalCount()

                    Q_d_list = []; p_d_list = []; acc_d_list = []
                    for d in dest_list:
                        if str(d) in self.labelQueues:
                            Q_d_list.append(self.labelQueues[str(d)].totalCount())
                            p_d_list.append(self.__estimator(label_src, label_dest))
                            acc_d_list.append(self.__checkTerminationBayes(label_dest)[1])

                    wght_l = Q_l - np.sum(q*p for q,p in zip(Q_d_list, p_d_list))
                    # max new labels
                    #inctv_l = (Q_l>0)*np.sum(k)/float(self.classifier_num)
                    # max Accuracy
                    inctv_l = (Q_l>0)*np.sum(acc*p for acc,p in zip(acc_d_list, p_d_list))

                    maxweightDict[(l, self.__listToStr(k))] = wght_l + self.V*inctv_l

                    #------------
                    if self.disp <= 1:
                        print 'dest_list', dest_list
                    #------------
                #else:
                #    print 'l,k, subset_src, subset_new, subset_diff:\n', l, k, subset_src,
                #    print subset_new, subset_diff
        #-------------------------
        #------display-------
        if self.disp <= 2:
            print 'maxweightDict:\n', maxweightDict;
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

        activeQs = self.labelQueues.keys()
        num_activeQs = len(activeQs)

        allowedSS = [self.__listToStr(s) for s in self.allowed_subset]
        num_allowedSS = len(allowedSS)

        num_edges = len(self.edges) # edges:
        #------display-------
        if self.disp <= 1:
            print '\nMaxweight:', maxweightDict
        #---------------------
        if num_nodes == 0:
            if self.totalCount() > 0:
                print '!!!Error!!!'
                print 'Nothing to schedule but Queue length:',self.totalCount()
                print 'The current state:'
                print
            return np.zeros(self.classifier_num), {}
        #---------------------------------
        ## Solving the Max Weighted Independent Set problem.
        x = cvx.Variable((num_nodes,1), boolean = True)
        y = cvx.Variable((num_allowedSS,1), boolean = True)
        #-----------------------------------------
        # constraints
        #-----------------------------------------
        # 1. Independent Set constraints (This is static)
        #-----constraint matrix----
        A = self.A
        #-----------------------------------------
        #2. At most one sample assigned to one subset
        Y = np.zeros((num_allowedSS, num_nodes))
        #print 'nodes:', [n[1] for n in nodes]
        #print 'allowedSS:', allowedSS

        for i, s in enumerate(allowedSS):
            jp = [jpp for jpp, n in enumerate(nodes) if n[1] == s]
            Y[i,jp] = 1
        #-------------------------------------------
        # 3. At most Q_l subsets assigned to labelqueue l
        B = np.zeros((num_activeQs, num_nodes))
        b = np.zeros((num_activeQs, 1))

        for i, activeQ in enumerate(activeQs):
            jp = [jpp for jpp, n in enumerate(nodes) if n[0] == activeQ];
            B[i,jp] = 1; b[i] = self.labelQueues[activeQ].totalCount()
        #-----------------------------------------
        #print '|Y|_1:', np.linalg.norm(Y,1)#, 'A:', A, 'B:', B, 'b:', b
        if num_edges:
            constraints = [A*y <= 1, 0<=y, y<=1, Y*x <= y, B*x <= b,  0<=x, x<=1]
        else:
            constraints = [0<=y, y<=1, Y*x <= y, B*x <= b,  0<=x, x<=1]
        #-----------------------------------------
        # objective
        #-----------------------------------------
        w = np.array([maxweightDict[n] for n in nodes])
        if self.disp <= 1:
            print 'weight data (n,w):',[(n, w[i])
                                     for i,n in enumerate(nodes)]
        # Form objective
        obj = cvx.Maximize(w.T*x)
        #-----------------------------------------
        # Form and solve problem
        #-----------------------------------------
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver = cvx.ECOS_BB)  # Returns the optimal value
        #----------------------------------------
        y_val = np.array(y.value)
        x_val = np.array(x.value)
        if self.disp <=1:
            print("status:", prob.status)
            print 'max y, min y:', np.max(y_val), np.min(y_val)
            print 'max Ay, min Ay:', np.max(np.dot(A,y_val)), np.min(np.dot(A,y_val))
            print 'max x, min x:', np.max(x_val), np.min(x_val)
            print 'max Yx-y, min Yx - y:', np.max(np.dot(Y,x_val)- y_val), np.min(np.dot(Y,x_val) - y_val)
            print 'max Bx-b, min Bx - b:', np.max(np.dot(B,x_val)- b), np.min(np.dot(B,x_val) - b)
        #-----------------------------------------
        try:
            node_active = [nodes[i] for i, t in enumerate(x_val) if t > 0.5]
            #print node_active
            if (self.totalCount() > 0) and (len(node_active) == 0):
                node_id = np.argmax(w)
                del node_active
                node_active = [nodes[node_id]]
                #if self.disp <=1:
                print '!!!Error!!!'
                print 'Nothing to schedule but Queue length:',self.totalCount()
                print("status:", prob.status)
                print("optimal var", x_val)
                print('Maxweight:[(label,subset):weight]')
                print maxweightDict
                print('Queues')
                print [(k,v.queue) for k, v in self.labelQueues.items()]
                print
                print '** New node_active:', node_active
        except Exception as e:
            #------------
            print '!!!Error!!!!!'
            print e
            print("status:", prob.status)
            print("optimal value", prob.value)
            print("optimal var", x_val)
            print('Maxweight [(label, subset): weight]')
            print maxweightDict
            print('Queues')
            print self.labelQueues
            print
            #------------
            return np.zeros(self.classifier_num), []
        #------display------------------------------
        if self.disp <= 1:
            print("status:", prob.status)
            print("optimal value", prob.value)
            print("optimal var", x_val)
        #--------------------------------------------
        # With the MWIS problem solved we need to form the schedule
        schedule = np.zeros(self.classifier_num)
        samplesOnTheFly = []
        #print "node_active:",node_active
        for node in node_active:
            schedule_old = np.copy(schedule)

            node_s = self.__strToList(node[1]) # subset of Classifiers
            node_l = self.__strToList(node[0]) # the label matched with the classifier

            # Get the sample Id from: lMax and source_lS
            sample_id = self.labelQueues[node[0]].queueExit()
            # remove the empty queues
            if self.labelQueues[node[0]].totalCount() == 0:
                del self.labelQueues[node[0]]
            # put the new sample in the samplesinTheFLy directory
            if sample_id is not None:
                samplesOnTheFly.append((sample_id, node_l)) # sample id, node_l= old_label
                #print 'id:', sample_id, 'subset:', node_s
                schedule += np.array(node_s)*int(sample_id)
                for i in range(self.classifier_num):
                    if node_s[i]:
                        self.classifierHist[i].setdefault(self.__listToStr(node_l), []).append(sample_id)

            #------display-------
            if self.disp <= -1:
                print('node:',node)
                print 'schedule old:', schedule_old, 'schedule new:', schedule
            #---------------------
            # Why 'sample_id == None' occurs?
            # More than one nodes may have the same source as the best choice in maxweight
            # In such situations, scheduling one node may make a non-empty queue empty

        #------display-------
        if self.disp <= -1:
            print 'MWIS:', node_active
            print 'sch:',schedule
            print 'samples:',samplesOnTheFly
        #--------------------
        return schedule, samplesOnTheFly

    def __explore_schedule(self):
        # pick a sample from label Subset label_0 if exists
            samplesOnTheFly  = [];
            label_0 = np.zeros(self.classifier_num).astype(int)
            id_0 = self.__listToStr(label_0)
            #----------
            # Get the sample Id from: lMax and source_lS
            sample_id = self.labelQueues[id_0].queueExit()
            # remove the empty queues
            if self.labelQueues[id_0].totalCount() == 0:
                del self.labelQueues[id_0]
            # put the new sample in the samplesinTheFLy directory
            if sample_id is not None:
                schedule = np.ones(self.classifier_num).astype(int)*int(sample_id)
                samplesOnTheFly.append((sample_id, label_0)) # sample id, node_l= old_label
            else:
                schedule = np.zeros(self.classifier_num).astype(int)
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
        id0 = self.__listToStr(np.zeros(self.classifier_num).astype(int))
        if explore and (id0 in self.labelQueues):
            schedule, samplesOnTheFly = self.__explore_schedule()
        else:
            schedule, samplesOnTheFly = self.__compute_schedule(maxweightDict)
        # schedule[i] = sample id scheduled to classifier i (0 = No sample)
        new_labels = self.classifierEnsemble.classify(schedule)
        new_labels_out = np.array([0]*self.classifier_num)
        # new_labels[i] = label coming from the i-th classifier
        # new_labels has to be list of len (num_classifier)
        #                or np.array of dim (num_classifier,1)
        #------------------------------
        # Update the queues now
        # sample_id >= 1 (0 is reserved for no entry to the classifier)
        sample_out = []
        #print 'schedule:', schedule, 'samplesOnTheFly:', samplesOnTheFly
        for sample_id in samplesOnTheFly:
            id, old_label = sample_id
            fresh_locs = [(schedule[i] == id) and (old_label[i] == 0) for i in range(self.classifier_num)]
            fresh_labels = np.multiply(new_labels, fresh_locs)

            #print 'id:',id, 'acquired_labels:', new_labels, 'fresh:', fresh_labels
            #-----display-------
            if self.disp <= 1:
                 print('old label:', old_label, 'fresh_locs:', fresh_locs, 'fresh_label:', fresh_labels)
            #---------------------
            new_label = old_label + fresh_labels
            if np.linalg.norm(old_label) == 0:
                new_labels_out = new_labels_out + fresh_labels
            # put the sample id with its label in the apt labelSubset queue

            if np.max(new_label) > self.num_classes:
                print '!!!Error!!!'
                print 'id:',id, 'old_label:',old_label
                print 'fresh_labels:', fresh_labels, 'fresh_locs:', fresh_locs
                print 'new_label:', new_label, 'max(new_label):', np.max(new_label)

            label_final, label_conf = self.__checkTerminationBayes(new_label, id)
            if label_final is not None:
                #-----display-------
                if self.disp <= 1:
                    print('new_label:', new_label, 'sample_id:%d'%id,
                    'label_final:%d'%label_final, 'label_conf:%.3f'%label_conf)
                #---------------------
                sample_out +=[(id, label_final, new_label, label_conf)]
            else:
                dest_id = self.__listToStr(new_label)
                if dest_id not in self.labelQueues:
                    self.labelQueues[dest_id] = labelQueue(new_label)
                self.labelQueues[dest_id].queueEntry(id)
        del samplesOnTheFly # nothing on the fly
        return sample_out, schedule, new_labels_out


    def newArrival(self, list_new_samples):
        # list_new_samples: list of new samples (id of each sample)
        label_0 = np.zeros(self.classifier_num).astype(int)
        # put the new arrivals in label_0 queue
        # --if label_0 queue not present create it
        id_0 = self.__listToStr(label_0)
        if id_0 not in self.labelQueues:
            self.labelQueues[id_0] = labelQueue(label_0)
        for entry in list_new_samples:
            self.labelQueues[id_0].queueEntry(entry)
