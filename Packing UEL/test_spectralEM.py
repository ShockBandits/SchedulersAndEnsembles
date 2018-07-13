
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import import_ipynb
from fakeENS import fakeENS
from spectralEM import spectralEM
from scipy.stats import entropy


# In[3]:


def str_arr(arr):
    return " ".join("%.2f"%x for x in arr)
#--------------------------------------------
def str_mat(mat):
    return "\n".join(str_arr(arr) for arr in mat) 


# In[4]:


num_classifier = 5
num_class = 3

ph_const = 0.1
ph_true = np.random.rand(num_class); 
ph_true = (ph_const + ph_true)/sum(ph_const + ph_true)

siz = int(1e4)
real_labels = np.random.choice(range(num_class), p = ph_true, size = siz)
ph_emp = np.bincount(real_labels)/ float(len(real_labels))

E = fakeENS(num_classifier, num_class, real_labels)
C_real = E.getConfMat()


# In[5]:


S = spectralEM(num_classifier, num_class, 
               maxiter = 100, num_init = 200, thres_const = 5e-4, disp = False)

print 'ph_true:%s \nph_emp:%s'%(str_arr(ph_true), str_arr(ph_emp))

group = S.group 
groupConfMat = {i: np.mean([C_real[j] for j in range(num_classifier) if group[i][j]], axis = 0)  for i in range(3)}
kappa = np.min([np.min([[groupConfMat[j][l,l] - groupConfMat[j][l,c] for c in range(num_class) if c !=l] 
                      for l in range(num_class)]) for j in range(3)])


barD = np.min([[np.mean([entropy(C_real[i][l,:], C_real[j][l,:]) 
                         for i in range(num_classifier)]) 
                for c in range(num_class) if c !=l] 
               for l in range(num_class)])

for i in range(3): print "Group %d:\n%s"%(i,str_mat(groupConfMat[i]));
print '\nkappa:%.3f, barD:%.3f'%(kappa, barD)

print 'Initial eigenvectors:\n%s'%str_mat(S.v_arr)


# In[12]:


E.reshuffle()
S.reset()

update_period = 500
est_period = 2*update_period
new_data = []
disp = True
errp_best = 100


for i in range(siz):
    schedule = (i+1)*np.ones(num_classifier).astype(int)
    labels = E.classify(schedule) - 1
    #print 'index:', i, 'real label:',real_labels[i], 'classifier labels:', labels
    new_data += [labels.astype(int)]
    if np.mod(i, update_period) == 0 and i>0: 
        S.update(new_data); 
        #-----------------------
        if np.mod(i, est_period) == 0 and i >0: 
            S.updateParamsSpectral('check')
        #-----------------------
        S.updateParamsEM(new_data, 10);
        new_data = []
    #------------------
        errp = np.linalg.norm(S.ph_est_avg - ph_true, 1)
        err = [np.linalg.norm(C_real[k]- S.conf_mat[k], 1) for k in range(num_classifier)]
        if errp_best > errp:
            ph_est_best = S.ph_est_avg.copy()
            conf_mat_best = S.conf_mat.copy()
            errp_best = errp
        #-----------------------------------------
        if disp:
            print '---------------------------'
            print 'Num samples:', S.num_data
            print 'Total L1 error in ph true: %.3f'%errp
            print 'Total L1 error in Conf Matrices: %.3f'%np.sum(err)
            print '---------------------------'
#------------------------------------------------------


# In[13]:


print 'ph_est:%s \nph_true:%s \nph_emp:%s'%(str_arr(S.ph_est_avg), str_arr(ph_true), str_arr(ph_emp))

for k in range(num_classifier):
    print '---------------------------'
    print 'Classifier:%d'%k
    print 'Confusion Matrix Final Estimate'
    print np.round(S.conf_mat[k],3)
    print 'Confusion Matrix Real'
    print np.round(C_real[k],3)
print '---------------------------'


# In[11]:


print 'ph_est:%s \nph_true:%s \nph_emp:%s'%(str_arr(ph_est_best), str_arr(ph_true), str_arr(ph_emp))
for k in range(num_classifier):
    print '---------------------------'
    print 'Classifier:%d'%k
    print 'Confusion Matrix Best Estimate'
    print np.round(conf_mat_best[k],3)
    print 'Confusion Matrix Real'
    print np.round(C_real[k],3)
print '---------------------------'

