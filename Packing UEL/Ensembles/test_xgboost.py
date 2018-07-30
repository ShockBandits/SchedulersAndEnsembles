from Cifar10.Classifiers.XGBoost import *

xgbc = XGBoost_Classifier("XGBC_0.cfg")
xgbc.create()
xgbc.get_train_data('data_batch_1')
xgbc.get_test_data('test_batch')
xgbc.fit()
xgbc.get_metrics()
xgbc.print_acc()
xgbc.print_conf_matrix('train')
xgbc.print_conf_matrix('test')
xgbc.save()

xgbc1 = XGBoost_Classifier("XGBC_0.cfg")
xgbc1.read()
xgbc1.print_acc()
xgbc1.print_conf_matrix('train')
xgbc1.print_conf_matrix('test')

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_xgboost.py 
Fitting XGBC_0_test
/home/innovationcommons/.local/lib/python2.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  if diff:
/home/innovationcommons/.local/lib/python2.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  if diff:
/home/innovationcommons/.local/lib/python2.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  if diff:
/home/innovationcommons/.local/lib/python2.7/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  if diff:
Training Acc: 0.4969
Testing Acc: 0.3736

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.534  0.051  0.036  0.014  0.017  0.021  0.027  0.033  0.221  0.047
autom  0.049  0.554  0.021  0.036  0.044  0.029  0.028  0.029  0.062  0.149
bird   0.099  0.046  0.310  0.036  0.144  0.041  0.194  0.057  0.049  0.024
cat    0.058  0.036  0.071  0.286  0.097  0.122  0.188  0.045  0.026  0.070
deer   0.056  0.020  0.074  0.027  0.450  0.055  0.190  0.057  0.042  0.028
dog    0.029  0.033  0.066  0.070  0.110  0.467  0.110  0.046  0.032  0.036
frog   0.026  0.035  0.050  0.020  0.090  0.042  0.683  0.030  0.006  0.017
horse  0.051  0.032  0.037  0.030  0.118  0.062  0.087  0.449  0.043  0.092
ship   0.124  0.064  0.008  0.036  0.015  0.026  0.035  0.018  0.600  0.074
truck  0.065  0.105  0.013  0.017  0.017  0.021  0.042  0.028  0.053  0.638 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.488  0.036  0.056  0.023  0.041  0.014  0.034  0.029  0.228  0.051
autom  0.067  0.408  0.023  0.037  0.041  0.042  0.062  0.032  0.071  0.217
bird   0.113  0.048  0.195  0.053  0.197  0.057  0.193  0.061  0.052  0.031
cat    0.072  0.045  0.074  0.131  0.109  0.171  0.224  0.067  0.035  0.072
deer   0.062  0.026  0.102  0.023  0.326  0.060  0.269  0.070  0.035  0.027
dog    0.058  0.035  0.089  0.097  0.137  0.304  0.126  0.080  0.040  0.034
frog   0.028  0.040  0.071  0.042  0.133  0.039  0.560  0.034  0.017  0.036
horse  0.061  0.031  0.057  0.050  0.130  0.092  0.088  0.314  0.053  0.124
ship   0.139  0.105  0.015  0.035  0.017  0.030  0.032  0.024  0.516  0.087
truck  0.084  0.149  0.012  0.031  0.028  0.024  0.039  0.044  0.095  0.494 

Saved to  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/XGBC_0_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/XGBC_0_test.pkl
Training Acc: 0.4969
Testing Acc: 0.3736

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.534  0.051  0.036  0.014  0.017  0.021  0.027  0.033  0.221  0.047
autom  0.049  0.554  0.021  0.036  0.044  0.029  0.028  0.029  0.062  0.149
bird   0.099  0.046  0.310  0.036  0.144  0.041  0.194  0.057  0.049  0.024
cat    0.058  0.036  0.071  0.286  0.097  0.122  0.188  0.045  0.026  0.070
deer   0.056  0.020  0.074  0.027  0.450  0.055  0.190  0.057  0.042  0.028
dog    0.029  0.033  0.066  0.070  0.110  0.467  0.110  0.046  0.032  0.036
frog   0.026  0.035  0.050  0.020  0.090  0.042  0.683  0.030  0.006  0.017
horse  0.051  0.032  0.037  0.030  0.118  0.062  0.087  0.449  0.043  0.092
ship   0.124  0.064  0.008  0.036  0.015  0.026  0.035  0.018  0.600  0.074
truck  0.065  0.105  0.013  0.017  0.017  0.021  0.042  0.028  0.053  0.638 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.488  0.036  0.056  0.023  0.041  0.014  0.034  0.029  0.228  0.051
autom  0.067  0.408  0.023  0.037  0.041  0.042  0.062  0.032  0.071  0.217
bird   0.113  0.048  0.195  0.053  0.197  0.057  0.193  0.061  0.052  0.031
cat    0.072  0.045  0.074  0.131  0.109  0.171  0.224  0.067  0.035  0.072
deer   0.062  0.026  0.102  0.023  0.326  0.060  0.269  0.070  0.035  0.027
dog    0.058  0.035  0.089  0.097  0.137  0.304  0.126  0.080  0.040  0.034
frog   0.028  0.040  0.071  0.042  0.133  0.039  0.560  0.034  0.017  0.036
horse  0.061  0.031  0.057  0.050  0.130  0.092  0.088  0.314  0.053  0.124
ship   0.139  0.105  0.015  0.035  0.017  0.030  0.032  0.024  0.516  0.087
truck  0.084  0.149  0.012  0.031  0.028  0.024  0.039  0.044  0.095  0.494 
'''
