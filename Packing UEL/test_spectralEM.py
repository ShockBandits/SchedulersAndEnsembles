
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


from fakeENS import fakeENS
from spectralEMNew import spectralEM
from scipy.stats import entropy


# In[3]:


def str_arr(arr):
    return " ".join("%.2f"%x for x in arr)
#--------------------------------------------
def str_mat(mat):
    return "\n".join(str_arr(arr) for arr in mat)


# In[4]:

#------------INputs------------------------------
init_update_after = 1000
update_period = 100
Spectral2EM = 12
est_period = Spectral2EM*update_period
num_Spectral_updates = 20
siz = int(min(1e6, num_Spectral_updates*est_period))
#-------------------------------------------------------------
num_classifier = 10
num_class = 3
#-------------------------------------------------------------
ph_const = 0.1
ph_true = np.random.rand(num_class);
ph_true = (ph_const + ph_true)/sum(ph_const + ph_true)
#---------------------------------------------------------------
real_labels = np.random.choice(range(num_class), p = ph_true, size = siz)
ph_emp = np.bincount(real_labels)/ float(len(real_labels))
print 'ph_true:%s \nph_emp:%s'%(str_arr(ph_true), str_arr(ph_emp))
disp = True
#---------------------Spectral Estimator------------------
num_EM_updates = 10
maxiter = 20
num_init = 200
EM_data_siz = 2000
#---------------------CLassifiers and Spectral Estiamtor----------------------------
E = fakeENS(num_classifier, num_class, real_labels)
C_real = E.getConfMat()
S = spectralEM(num_classifier, num_class,
               maxiter = 50, num_init = 200, EM_data_siz= EM_data_siz,
               thres_const = 5e-4, disp = False)
#-------------------Basic Stats-------------------
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

#-------Initialize------------
E.reshuffle()
S.reset()
new_data = []
#-----------------------------
errp_best = 100
err_list = []

for i in range(siz):
    schedule = (i+1)*np.ones(num_classifier).astype(int)
    labels = E.classify(schedule) - 1
    #print 'index:', i, 'real label:',real_labels[i], 'classifier labels:', labels
    new_data += [labels.astype(int)]
    if np.mod(i, update_period) == 0 and i>init_update_after:
        S.update(new_data);
        #-----------------------
        if np.mod(i, est_period) == 0:
            S.updateParamsSpectral('check', MaxIter = maxiter, num_init = num_init)
            spectralUpdateflag = True
        else:
            spectralUpdateflag = False
        #-----------------------
        S.updateParamsEM(num_EM_updates);
        new_data = []
    #------------------
        errp = np.linalg.norm(S.ph_est_avg - ph_true, 1)
        err = [np.linalg.norm(C_real[k]- S.conf_mat[k], 1) for k in range(num_classifier)]
        if errp_best > errp:
            ph_est_best = S.ph_est_avg.copy()
            conf_mat_best = S.conf_mat.copy()
            errp_best = errp
        err_list += [[errp, np.sum(err), spectralUpdateflag]]
        #-----------------------------------------
        if disp:
            print '---------------------------'
            print 'Spectral update?',spectralUpdateflag
            print 'Num samples:', S.num_data
            print 'Total L1 error in ph true: %.3f'%errp
            print 'Total L1 error in Conf Matrices: %.3f'%np.sum(err)
#------------------------------------------------------


# In[13]:
print '---------------------------'
print 'Final Estimate'
print '---------------------------'

print 'ph_est:%s \nph_true:%s \nph_emp:%s'%(str_arr(S.ph_est_avg), str_arr(ph_true), str_arr(ph_emp))

for k in range(num_classifier):
    print '---------------------------'
    print 'Classifier:%d'%k
    print 'Confusion Matrix (Final)'
    print np.round(S.conf_mat[k],3)
    print 'Confusion Matrix Real'
    print np.round(C_real[k],3)
print '---------------------------'


# In[11]:

print '---------------------------'
print 'Best Estimate'
print '---------------------------'


print 'ph_est:%s \nph_true:%s \nph_emp:%s'%(str_arr(ph_est_best), str_arr(ph_true), str_arr(ph_emp))
for k in range(num_classifier):
    print '---------------------------'
    print 'Classifier:%d'%k
    print 'Confusion Matrix (Best)'
    print np.round(conf_mat_best[k],3)
    print 'Confusion Matrix Real'
    print np.round(C_real[k],3)
print '---------------------------'

#----Plot the Error----------------------------

plt.plot(   range(len(err_list)), [e[0] for e in err_list],'b-',
            range(len(err_list)),[e[1] for e in err_list] , 'r-',
         )
for i in range(len(err_list)):
    if err_list[i][2]:
        plt.axvline(x=i)
plt.legend(['p_true', 'conf mat'])
plt.title('Evolution of Estimation Error')
plt.xlabel('timeslot')
plt.show()
