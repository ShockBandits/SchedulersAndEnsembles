from Cifar10_Ensemble.Cifar10_Classifiers.RandomForest import *

rfc = RandomForest_Classifier("RFC_0.cfg")
rfc.create()
rfc.get_train_data('data_batch1')
rfc.get_test_data('test_batch')
rfc.fit()
rfc.get_metrics()
rfc.print_acc()
rfc.print_conf_matrix('train')
rfc.print_conf_matrix('test')
rfc.save()

rfc1 = RandomForest_Classifier("RFC_0.cfg")
rfc1.read()
rfc1.print_acc()
rfc1.print_conf_matrix('train')
rfc1.print_conf_matrix('test')

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_random_forest.py 
Fitting RFC_0_test
Training Acc: 0.9998
Testing Acc: 0.3725

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.999  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
autom  0.000  0.999  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.001
bird   0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
cat    0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000  0.000
deer   0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000
dog    0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000
frog   0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000
horse  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000
ship   0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000
truck  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.508  0.053  0.076  0.028  0.027  0.020  0.031  0.033  0.178  0.046
autom  0.061  0.407  0.043  0.051  0.041  0.037  0.052  0.034  0.081  0.193
bird   0.127  0.050  0.307  0.087  0.142  0.071  0.104  0.064  0.031  0.017
cat    0.069  0.049  0.128  0.206  0.096  0.161  0.132  0.064  0.034  0.061
deer   0.057  0.029  0.196  0.082  0.329  0.052  0.143  0.059  0.027  0.026
dog    0.050  0.053  0.136  0.185  0.085  0.284  0.085  0.067  0.032  0.023
frog   0.026  0.054  0.148  0.096  0.126  0.051  0.422  0.035  0.008  0.034
horse  0.065  0.057  0.083  0.085  0.123  0.080  0.057  0.328  0.033  0.089
ship   0.132  0.100  0.034  0.050  0.025  0.035  0.012  0.026  0.512  0.074
truck  0.078  0.197  0.036  0.048  0.023  0.025  0.025  0.039  0.107  0.422 

Saved to  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RFC_0_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RFC_0_test.pkl
Training Acc: 0.9998
Testing Acc: 0.3725

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.999  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
autom  0.000  0.999  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.001
bird   0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
cat    0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000  0.000
deer   0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000  0.000
dog    0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000  0.000
frog   0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000  0.000
horse  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000  0.000
ship   0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000
truck  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.508  0.053  0.076  0.028  0.027  0.020  0.031  0.033  0.178  0.046
autom  0.061  0.407  0.043  0.051  0.041  0.037  0.052  0.034  0.081  0.193
bird   0.127  0.050  0.307  0.087  0.142  0.071  0.104  0.064  0.031  0.017
cat    0.069  0.049  0.128  0.206  0.096  0.161  0.132  0.064  0.034  0.061
deer   0.057  0.029  0.196  0.082  0.329  0.052  0.143  0.059  0.027  0.026
dog    0.050  0.053  0.136  0.185  0.085  0.284  0.085  0.067  0.032  0.023
frog   0.026  0.054  0.148  0.096  0.126  0.051  0.422  0.035  0.008  0.034
horse  0.065  0.057  0.083  0.085  0.123  0.080  0.057  0.328  0.033  0.089
ship   0.132  0.100  0.034  0.050  0.025  0.035  0.012  0.026  0.512  0.074
truck  0.078  0.197  0.036  0.048  0.023  0.025  0.025  0.039  0.107  0.422 

innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ 
'''
