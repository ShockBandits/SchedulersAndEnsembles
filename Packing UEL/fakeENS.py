
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


class fakeENS:
    def __init__(self, num_classifier, num_class,
                 real_labels = None, confMat = None, alpha = None,
                 excess_acc = np.array([0.8, 0.5, 0.3])):
        '''
        num_classifier: number of classifiers
        num_class: number of classes
        real_labels: stream of the real labels
        alpha: minimum accuracy of the classifiers
        excess_acc: excess accuracy of the classifiers
        '''
        self.num_class = num_class
        self.num_classifier = num_classifier
        #minimum accuracy per worker
        if confMat is None:
            if alpha is None:
                alpha_p = [0.5, 0.3, 0.2] # probability of choosing alpha from alpha_arr
                self.alpha = np.random.choice(excess_acc, p = alpha_p,
                                              size = (num_classifier, num_class))
            else:
                self.alpha = alpha
            #------------------------
            # generate the confusion matrices
            self.confMat = {i: self.__genConfMat(self.alpha[i,:]) for i in range(self.num_classifier)}
        else:
            self.confMat = confMat.copy()
        # real labels storage
        self.real_labels = real_labels
        #--------used to fix the classification-------
        self.a = np.random.randint(100);
        self.b = np.random.randint(1000, 2000);

    def reshuffle(self):
        self.a = np.random.randint(100);
        self.b = np.random.randint(1000, 2000);

    def newSamples(self, real_labels):
        # real labels storage
        self.real_labels = real_labels

    def getConfMat(self):
        return self.confMat

    def __genConfMat(self, alpha = None):
    # k is the size of the confusion matrix
    # alpha is the lower limit for the diagonal entry
        if alpha is None:
            alpha = 0.51*np.ones(self.num_class)
        rho = 1e-6
        normP = lambda x: np.maximum(x, rho)/np.sum(np.maximum(x, rho))
        k = self.num_class
        C = np.array([np.random.rand(k) for i in range(k)])
        for i in range(k):
            C[i,i] = alpha[i] + np.max(C[i,:]);
            C[i,:] = normP(C[i,:])
        return C

    def classify(self, schedule):
        #print 'sch_input:',schedule
        new_labels = np.zeros(self.num_classifier)
        #print 'schedule:', schedule
        for i, sample_id in enumerate(schedule):
            #----------------------------
            if sample_id >= 1:
                # Seed to make labelling deterministic
                np.random.seed(int(self.a*sample_id+self.b*i))
                h = self.real_labels[int(sample_id - 1)]
                l = np.random.choice(range(self.num_class), p = self.confMat[i][int(h),:])
                new_labels[i] = l+1 # label 0 resereved for null label
                #print 'sample id:', sample_id, 'classifier:', i, 'seed:', seed, 'label:', l+1
        return new_labels
