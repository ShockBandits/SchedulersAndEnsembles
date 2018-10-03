
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from time import time
from itertools import permutations, product
from munkres import Munkres
import random


# In[1]:


class spectralEM:
    '''
    Maintains the Tensors and Matrices used for parameter estimation.
    The parameters are the confusion matrix and the fraction of the true labels.

    Each new data is an array of labels from all classifiers for a specific sample.
    It needs to send each sample to all the classifiers in the explore phase.
    '''
    #--------------------------------------------
    def __init__(self, num_classifiers, num_classes, maxiter = 50,
                 num_init = 5, thres_const = 1e-3, EM_data_siz= 1000, disp = True):
        self.num_classifier = num_classifiers
        self.num_class = num_classes
        # self.group: dict| key: group number, value: list of classifiers in the group
        # self.groupid[i]: group of i-th classifier
        self.group, self.groupid = self.__getGroup()
        # permutation of the 3 groups
        self.perm = [(1,2,0), (0,1,2), (2,0,1)]
        #--------------------------Tensor Decomopsition params
        self.MaxIter = maxiter
        self.num_init = num_init
        self.thres_iter = thres_const*np.sqrt(self.num_class)
        #-------------------------EM parameters---------
        self.EM_data_siz = EM_data_siz
        self.EM_data = []
        #-----------------display
        self.disp = disp
        #------------------from reset
        self.reset()
    #-----------------------------------------------

    def reset(self):
        # the running averages maintained in self.M dictionary
        # This is the summary of data sufficient to compute hidden parameters
        L = [(0,1), (1,2), (2,0)] + range(self.num_classifier)
        # Initialization of parameters
        self.M = {str(k): np.zeros((self.num_class, self.num_class)) for k in L}
        self.M['(0,1,2)'] = np.zeros((self.num_class, self.num_class, self.num_class))
        self.num_data = 0 # number of samples used
        self.classifier_count = np.zeros(self.num_classifier)
        #-------conf matrices---------------------
        self.ph_est_avg = np.ones(self.num_class)/float(self.num_class)
        M = np.ones((self.num_class, self.num_class))/float(self.num_class)
        expert = 0.1
        self.conf_mat = {i: expert*np.eye(self.num_class)+(1-expert)*np.copy(M)
                         for i in range(self.num_classifier)}
        #-----initialize candidate eigenvalue--------
        self.v_arr = np.zeros((self.num_class, self.num_class))
        for i in range(self.num_class):
            v = np.random.randn(self.num_class); v = v/(np.linalg.norm(v)+1e-9)
            self.v_arr[:,i] = v.copy()

    #------------------------------------------------
    # The auxiliary/helper Tensor Algebra functions
    def __str_arr(self, arr):
        return " ".join("%.2f"%x for x in arr)
    #--------------------------------------------
    def __str_mat(self, mat):
        return "\n".join(self.__str_arr(arr) for arr in mat)
    #--------------------------------------------
    def __oneHot(self, arr, dim):
        # make all the entries in arr (categorical) to one-hot arrays
        arr = arr.astype(int)
        res = np.zeros((len(arr), dim))
        for i, e in enumerate(arr):
            if e >=0:
                res[i, e] = 1.0
        #------------------
        checked = True
        if not checked:
            print 'arr:', arr
            print 'res:', res
        #------------------
        return res
    #--------------------------------------------
    def __normP(self, x):
        return np.maximum(x, 1e-9)/np.sum(np.maximum(x, 1e-9))
    #--------------------------------------------
    def __tensorProdk(self, a,k):
        res = a
        for _ in range(k-1):
            res = np.tensordot(res, a, axes = 0)
        return res
    #--------------------------------------------
    def __tensorwhiten(self, T,W):
        # T(W,W,W)
        I,J,K = T.shape
        T_new = np.zeros((I,J,K))
        for i,j,k in product(range(I), range(J), range(K)):
            T_new[i,j,k] = np.sum([ T[i_p,j_p,k_p]*W[i_p,i]*W[j_p,j]*W[k_p,k]
                                   for i_p,j_p,k_p in product(range(I), range(J), range(K))])
        return T_new
    #--------------------------------------------
    def __tensormult1(self, T, v):
        # T(I,v,v)
        I,J,K = T.shape
        M_new = np.zeros((I, J))
        for i, j in product(range(I), range(J)):
            M_new[i,j] = np.sum([T[i,j,k_p]*v[k_p]
                               for k_p in range(K)])
        return M_new
    #--------------------------------------------
    def __tensormult2(self, T, v):
        # T(I,v,v)
        I,J,K = T.shape
        v_new = np.zeros(I)
        for i in range(I):
            v_new[i] = np.sum([T[i,j_p,k_p]*v[j_p]*v[k_p]
                               for j_p,k_p in product(range(J), range(K))])
        return v_new
    #--------------------------------------------
    def __tensormult3(self, T, v):
        # T(v,v,v)
        I,J,K = T.shape
        val = np.sum([T[i_p,j_p,k_p]*v[i_p]*v[j_p]*v[k_p]
                      for i_p,j_p,k_p in product(range(I), range(J), range(K))])
        return val
    #--------------------------------------------
    # One time functions
    def __getGroup(self):
        g = {}
        while len(g.keys()) < 3:
            groups = np.random.choice([0,1,2], p = [1/3.0, 1/3.0, 1/3.0], size = self.num_classifier)
            g ={i: list(groups ==i) for i in range(3) if np.sum(groups ==i) > 0}
        return g, groups
    #-----------------------------------------------
    # Update the database with new_data
    def update(self, new_data):
        # new_data: list of samples
        # new_data[j][i]: i-th classifier, j-th time slot
        #----update the EM data set---------------
        self.EM_data += new_data
        if len(self.EM_data) > self.EM_data_siz:
            random.shuffle(self.EM_data)
            self.EM_data = self.EM_data[:self.EM_data_siz]
        #-----------------------------------------
        oh_data = np.array([self.__oneHot(x, self.num_class) for x in new_data])
        # oh_data[j][i][l]:i-th classifier, j-th time slot, l-th class
        num_new = len(new_data); num_tot = self.num_data+num_new;
        # print 'num_new:', num_new, 'num_old:', self.num_data, 'num_tot', num_tot

        # aggreagte data accross groups
        Z = {i: np.mean(oh_data[:,self.group[i],:], axis = 1) for i in [0,1,2]}
        #-------
        checked = True


        if not checked: print 'Passed part I';
        #-------------------------------
        # Update the self.M dictionary with the new data
        for tup in [(0,1), (1,2), (2,0)]:
            newM = np.array([np.tensordot(t1, t2, axes = 0)
                             for t1, t2 in zip(Z[tup[0]], Z[tup[1]])])
            self.M[str(tup)] = (self.M[str(tup)]*self.num_data
                                + np.sum(newM, axis = 0))/float(num_tot)

        if not checked: print 'Passed part II';
        #------------------------------------
        for i in range(self.num_classifier):
            g = self.groupid[i]
            a = np.mod(g+1, 3)
            newM = np.array([np.tensordot(t1, t2, axes = 0)
                             for t1, t2 in zip(oh_data[:,i,:], Z[a])])
            self.M[str(i)] = (self.M[str(i)]*self.num_data
                              + np.sum(newM, axis = 0))/float(num_tot)

        if not checked: print 'Passed part III';
        #-------------------------------------
        newM = np.array([np.tensordot(np.tensordot(t1, t2, axes = 0), t3, axes = 0)
                             for t1, t2, t3 in zip(Z[0], Z[1], Z[2])])
        self.M['(0,1,2)'] = (self.M['(0,1,2)']*self.num_data
                              + np.sum(newM, axis = 0))/float(num_tot)

        if not checked: print 'Passed part IV';
        #-------------------------------------
        self.num_data = num_tot
        for new_d in new_data:
            self.classifier_count += np.all(np.array(new_d)>0)
        #-------------------------------------
        return
    #-----------------Parameter Estimation----------
    # The data is summarized in self.M which is used in Parameter Estimation
    # Recovery functions (One shot recovery from updated moments)
    def __getGroupMoments(self):
        # Returns the Group Symmetric Moments
        #-----------------------------
        fSwap = lambda x,y: (self.M[str((x,y))] if ((x,y) in [(0,1), (1,2), (2,0)])
                             else self.M[str((y,x))].T)
        #-----------------------------
        M2 = {}; M3 = {};
        checked = True

        if not checked: print 'In getGroupMoments';
        #-----------------------------
        for p in self.perm:
            a, b, c = p
            Mcb = fSwap(c,b); Mab = fSwap(a,b);
            Mca = fSwap(c,a); Mba = fSwap(b,a);
            Mabc = np.moveaxis(self.M['(0,1,2)'], (0,1,2), p)

            if not checked: print 'Passed Part I in getGroupMoments';
            #-----------------------------
            try:
                Mb = np.dot(Mcb, np.linalg.pinv(Mab));
                Ma = np.dot(Mca, np.linalg.pinv(Mba));
            except:
                print('Singular Matrix Mab or Mba')
                print "Mab\n:", Mab
                print "Mba\n:", Mba
                return None, None

            if not checked: print 'Passed Part II in getGroupMoments';
            #-----------------------------
            M2[c] = np.dot(np.dot(Mb, Mab), Ma.T);
            M3[c] = np.tensordot(np.tensordot(Mb, Mabc, axes = 1), Ma, axes = [[1], [1]])

            if not checked: print 'Passed Part III in getGroupMoments';
        #-----------------------------
        return M2, M3

    def __getGroupConf(self, mode = 'nocheck', restart = 'all',
                       Maxiter = None, num_init = None):
        # Returns estimated group Confidence Matrices
        # Returns estimated true probabilities
        #---------------------Parameters
        rank = self.num_class;
        checked = True
        if Maxiter is None: Maxiter = self.MaxIter;
        if num_init is None: num_init = self.num_init;
        thres_iter = self.thres_iter
        disp = self.disp

        if not checked: print 'In getGroupConf';
        #---------------------Get the moements
        M2, M3 = self.__getGroupMoments()

        if not checked: print 'Passed Part I in getGroupConf';
        #----------------------Initialiaztion
        W ={}; mu_est = {}; ph_est = {}
        #----------------Iteration over the groups
        for j in range(3):
            c = self.perm[j][2] # group c parameters
            if disp: print '**Group:(%d)**'%c;
            ## whitenning M3
            [U, l, V] = np.linalg.svd(M2[j]);
            L = np.diag(l)
            W = np.dot( U[:, :rank], np.sqrt(np.linalg.inv(L[:rank, :rank])) );
            M3W = self.__tensorwhiten(M3[j], W)
            #---------------------------------
            if not checked: print 'Passed Part II in getGroupConf'
            #---------------------------------
            ## tensor decomposition
            alpha_arr = np.zeros(rank)
            v_arr = np.zeros((rank, rank))
            #------
            for i in range(rank): # loop over eigenvalues
                #------------------------------------------------
                count_init = 0; break_expr = False; best_val = - 100
                # loop over initialization
                for _ in range(num_init):
                    # random initialization
                    v_old = np.random.randn(rank);
                    v_old = v_old/(np.linalg.norm(v_old)+1e-6);
                    # initialize other params
                    #----------power method iteration
                    for _ in range(Maxiter):
                        # one step tensor product
                        v_new = self.__tensormult2(M3W, v_old);
                        v_new = v_new/(np.linalg.norm(v_new)+1e-6);
                        v_old = v_new
                    #----------------------------------
                    new_val = self.__tensormult3(M3W, v_new)
                    if best_val < new_val:
                        alpha_arr[i] = new_val # eigenvalue
                        v_arr[:,i] = v_new # eigenvector
                # another round of power updates with the best point
                v_old = v_arr[:,i]
                for j in range(Maxiter):
                    v_new = self.__tensormult2(M3W, v_old); v_new = (v_new+1e-3)/np.linalg.norm(v_new+1e-3);
                    v_old = v_new
                alpha_arr[i] = self.__tensormult3(M3W, v_new) # final eigenvalue
                v_arr[:,i] = v_new # final eigenvector


                # deflate tensor
                M3W -= alpha_arr[i]*self.__tensorProdk(v_arr[:,i],3)
                self.v_arr[:,i] = v_arr[:,i].copy()
                if disp: print 'Tensor norm after %d-th eigenpair:%.3f'%(i, np.linalg.norm(M3W));

            if not checked: print 'Passed Part III in getGroupConf'
            #--------
            if disp: print 'Recovered eigenpairs:\n%s'%"\n".join('%.2f:[%s]'%(alpha_arr[i], self.__str_arr(v_arr[:,i])) for i in range(self.num_class));
            #----------------------------------------------------
            # FROM EIGENPAIRS TO THE ACTUAL PARAMETERS JOINTLY
            mu_est_int = np.zeros((rank, rank))
            ph_est_int = np.zeros(rank)
            #-----------------------------------------------------
            for i in range(self.num_class):
                # Unwhittening the vectors are stacked into columns for mu_est_int
                # alpha_arr[i] = 1/ sqrt(ph_est_int[i])
                ph_est_int[i] = alpha_arr[i]**(-2)
                # v_arr[:,i] = np.dot(W.T, sqrt(ph_est_int[i])*mu_est_int[i,:])
                mu_est_int[i,:] = self.__normP(alpha_arr[i]*np.dot(np.linalg.pinv(W.T), v_arr[:,i]))
                # Stack the eigenvector rowwise: mu_est_int[i,:] = E['one-hot o/p label'| true label= i]
            #-----------------------------------------------------
            # Finding the correct permutation of eigenpairs
            # permutation that maximizes diagonal entry
            #-----------------------------------------------------
            max_val = np.linalg.norm(mu_est_int,1)
            wght_mat = [[(max_val - mu_est_int[old,new]) for new in range(self.num_class)]
                        for old in range(self.num_class)]
            mun = Munkres(); mun_indexes = mun.compute(wght_mat)
            if disp: print 'mun_indexes: ',mun_indexes;
            ind = [0]*self.num_class
            for old, new in mun_indexes: ind[int(new)] = int(old)
            if disp: print 'New Order: %s'%self.__str_arr(ind);
            # rearranging the rows(?)
            mu_est[c] = mu_est_int[ind,:]
            ph_est[c] = self.__normP(ph_est_int[ind])
            #-----------------------------------------------------
            if disp: print 'mu_est[%d]:\n%s'%(c,self.__str_mat(mu_est[c]));
        #----------------------------------------------------
        if not checked: print 'Passed Part IV in getGroupConf';
        #-------------------------------------------------------------------
        ph_est_avg = np.mean([ph_est[j] for j in range(3)], axis = 0)
        ph_est_avg = (ph_est_avg+1e-3)/float(np.sum(ph_est_avg+1e-3))
        #------------------display results------------------
        if disp:
            for c in range(3):
                print 'Group:%d'%c
                print 'mu_est:\n%s'%(self.__str_mat(mu_est[c]));
                print 'ph_est:\n%s'%(self.__str_arr(ph_est[c]));
            print 'ph_est_avg:\n%s'%self.__str_arr(ph_est_avg);
        #-----------------------------
        return mu_est, ph_est_avg
    #---------------------------------------------------
    def __compute_q_hat(self, labels):
        # for a particular sample compute q_hat
        # labels[i,:]: the one hot encoding of label of classifier i
        # C: dictionary of confusion matrices
        arr = np.sum(np.dot(np.log(self.conf_mat[i]), labels[i,:])
                     for i in range(self.num_classifier))
        wghts = np.multiply(self.ph_est_avg, np.exp(arr-max(arr)))
        #wghts = np.exp(arr-max(arr)) #uniform prior
        q_hat = np.maximum(wghts, 1e-9)/np.sum(np.maximum(wghts, 1e-9))
        return q_hat
    #---------------------------------------------------
    #--------------------------------------------------
    def updateParamsEM(self, num_steps = 5):
        # new_data: list of samples
        # new_data[j][i]: i-th classifier, j-th time slot
        oh_data = np.array([self.__oneHot(x, self.num_class) for x in self.EM_data])
        # oh_data[j][i][l]:i-th classifier, j-th time slot, l-th class
        # Perform EM steps on the current parameters
        C_iter = self.conf_mat.copy()
        for _ in range(num_steps):
            # E-step
            q_hat_arr = np.array([self.__compute_q_hat(label) for label in oh_data])
            # M-step
            for i in range(self.num_classifier):
                mu_int = np.dot(q_hat_arr.T, oh_data[:,i,:] )
                # normalize the vectors accross rows
                C_iter[i] = np.divide(mu_int.astype(float), mu_int.sum(axis = 1, keepdims = True))
            # update conf_mat (Maximization)
            self.conf_mat = C_iter.copy()
            # update ph_est (gradient step)
            wght_k = np.mean(q_hat_arr,0)
            min_ph = np.min(self.ph_est_avg)
            ph_add = min_ph*np.divide(wght_k, self.ph_est_avg)
            ph_est_new = self.__normP(self.ph_est_avg + ph_add)
            self.ph_est_avg = ph_est_new

        return
    #---------------------------------------------------
    def updateParamsSpectral(self, mode = 'nocheck', restart = 'all',
                             MaxIter = None, num_init = None):
        # Returns the parameter to the main function
        # Specifically, returns the individual confidence Matrices
        # and the true probability vector over labels
        #-----------------------------
        checked = True
        disp = self.disp

        if not checked: print 'In getParams';

        mu_est, ph_est_avg = self.__getGroupConf(mode, restart, MaxIter, num_init)

        if not checked: print 'Passed Part I in getParams';
        #-----------------------------
        rank = self.num_class; C = {}
        for j in range(3):
            c = self.perm[j][2]; a = self.perm[j][0];

            P = np.dot(np.diag(ph_est_avg), mu_est[a].T)
            try: # P maybe singular
                P1 = np.linalg.inv(P)
            except: # This is a hacky fix
                return


            c_list = np.arange(self.num_classifier)[self.group[c]]

            for i in c_list.astype(int):
                P2 = self.M[str(i)]
                C[i] = np.dot(P2, P1)
                C[i] = np.array([self.__normP( np.maximum(1e-2, np.minimum(1,C[i][j,:])) )
                                                for j in range(self.num_class)])
                if disp: print 'Classifier %d:\n%s'%(i, self.__str_mat(C[i]));
        if not checked: print 'Passed Part II in getParams';
        #-----------------------------
        self.ph_est_avg = self.__normP(np.maximum(1e-2,ph_est_avg))
        self.conf_mat = C.copy()
        #------------------------------
        return
