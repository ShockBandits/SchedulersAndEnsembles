
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import import_ipynb
from fakeENS import fakeENS


# In[3]:


def str_arr(arr, mode = 'float'):
    if mode == 'int':
        return " ".join("%d"%x for x in arr)
    else:
        return " ".join("%.3f"%x for x in arr)

def averageAcc(p_true, C):
    return np.sum(C[i,i]*p for i, p in enumerate(p_true))    

def norm_row(M):
    row_sums = M.sum(axis=1).astype('float')
    new_M = M /row_sums[:, np.newaxis]
    return new_M


# In[4]:


num_class = 5
p_true = np.random.rand(num_class); p_true = p_true/ np.sum(p_true)

num_classifier = 7

num_tot_samp = 10000
stream_sample = np.arange(num_tot_samp)+1
stream_true_label = np.random.choice(num_class, num_tot_samp, p = p_true)


# In[8]:


# Replace with the actual Ensemble from Steven's github
ENS = fakeENS(num_classifier, num_class, stream_true_label)
conf_mat = ENS.getConfMat()
avg_acc = [averageAcc(p_true, conf_mat[i]) for i in range(num_classifier)]

print 'Min accuracy: %s'%str_arr(ENS.alpha)
print 'Avg accuracy: %s'%str_arr(avg_acc)


# In[9]:


conf_count = np.zeros((num_classifier, num_class, num_class))

for i in range(num_tot_samp):
    real_label = stream_true_label[i]
    schedule = stream_sample[i]*np.ones(num_classifier)
    labels = ENS.classify(schedule) - 1
    #print "real:", real_label, "labels:", labels
    for j in range(num_classifier): conf_count[j, real_label, int(labels[j])] += 1;


# In[10]:


# compute empirical confusion matrices
conf_mat_emp = {j: norm_row(conf_count[j,:,:]) for j in range(num_classifier)}
# compute the errors
err = np.sum(np.linalg.norm(conf_mat_emp[j] - conf_mat[j], 1)for j in range(num_classifier))
# display error
print 'Error in empirical vs true conf matrix:%.3f'%err

