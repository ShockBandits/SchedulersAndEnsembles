from Cifar10_Ensemble.Cifar10_Classifiers.SimpleCNN import *
scnn = SimpleCNN_Classifier("SCNN_0.cfg")
scnn.get_train_data('data_batch_1')
scnn.get_test_data('test_batch')
scnn.create()
scnn.fit()
scnn.get_metrics()
scnn.print_acc()
scnn.print_conf_matrix('train')
scnn.print_conf_matrix('test')
scnn.save()

scnn1 = SimpleCNN_Classifier("SCNN_0.cfg")
scnn1.read()
scnn1.print_acc()
scnn1.print_conf_matrix('train')
scnn1.print_conf_matrix('test')

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_simple_cnn.py 
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
Fitting SCNN_0
Epoch 1/1
10000/10000 [==============================] - 2s 210us/step - loss: 2.2002 - acc: 0.1757
Using real-time data augmentation.
Epoch 1/5
313/313 [==============================] - 3s 9ms/step - loss: 1.9433 - acc: 0.2849 - val_loss: 1.8155 - val_acc: 0.3638
Epoch 2/5
313/313 [==============================] - 2s 8ms/step - loss: 1.8146 - acc: 0.3394 - val_loss: 1.6986 - val_acc: 0.4017
Epoch 3/5
313/313 [==============================] - 3s 8ms/step - loss: 1.7444 - acc: 0.3601 - val_loss: 1.6309 - val_acc: 0.4188
Epoch 4/5
313/313 [==============================] - 3s 8ms/step - loss: 1.6705 - acc: 0.3880 - val_loss: 1.5716 - val_acc: 0.4375
Epoch 5/5
313/313 [==============================] - 3s 8ms/step - loss: 1.6188 - acc: 0.4041 - val_loss: 1.5211 - val_acc: 0.4528
10000/10000 [==============================] - 0s 36us/step
10000/10000 [==============================] - 0s 31us/step
Training Acc: 0.4624
Testing Acc: 0.4528

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.428  0.095  0.028  0.009  0.013  0.002  0.004  0.034  0.355  0.033
autom  0.024  0.691  0.002  0.004  0.000  0.000  0.028  0.024  0.139  0.089
bird   0.117  0.043  0.266  0.048  0.169  0.043  0.112  0.115  0.067  0.020
cat    0.028  0.042  0.074  0.282  0.054  0.139  0.167  0.119  0.031  0.063
deer   0.074  0.017  0.118  0.048  0.360  0.030  0.142  0.157  0.038  0.015
dog    0.020  0.037  0.089  0.163  0.052  0.279  0.105  0.201  0.023  0.031
frog   0.013  0.057  0.061  0.064  0.106  0.008  0.570  0.058  0.025  0.038
horse  0.030  0.049  0.028  0.037  0.080  0.058  0.058  0.575  0.021  0.064
ship   0.109  0.103  0.010  0.003  0.000  0.006  0.009  0.008  0.717  0.035
truck  0.016  0.261  0.005  0.013  0.003  0.001  0.040  0.033  0.178  0.450 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.426  0.083  0.038  0.012  0.009  0.002  0.016  0.035  0.348  0.031
autom  0.016  0.676  0.004  0.002  0.001  0.003  0.023  0.015  0.164  0.096
bird   0.099  0.040  0.283  0.060  0.151  0.040  0.120  0.117  0.069  0.021
cat    0.046  0.040  0.065  0.233  0.052  0.135  0.180  0.146  0.039  0.064
deer   0.057  0.028  0.131  0.050  0.355  0.019  0.152  0.159  0.039  0.010
dog    0.029  0.033  0.092  0.151  0.062  0.282  0.097  0.196  0.041  0.017
frog   0.005  0.050  0.068  0.044  0.091  0.009  0.592  0.071  0.019  0.051
horse  0.031  0.039  0.031  0.049  0.061  0.060  0.055  0.581  0.026  0.067
ship   0.111  0.086  0.013  0.010  0.001  0.004  0.010  0.014  0.707  0.044
truck  0.022  0.277  0.007  0.006  0.003  0.003  0.045  0.034  0.210  0.393 

Saved to  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/SCNN_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/SCNN_0.pkl
Training Acc: 0.4624
Testing Acc: 0.4528

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.428  0.095  0.028  0.009  0.013  0.002  0.004  0.034  0.355  0.033
autom  0.024  0.691  0.002  0.004  0.000  0.000  0.028  0.024  0.139  0.089
bird   0.117  0.043  0.266  0.048  0.169  0.043  0.112  0.115  0.067  0.020
cat    0.028  0.042  0.074  0.282  0.054  0.139  0.167  0.119  0.031  0.063
deer   0.074  0.017  0.118  0.048  0.360  0.030  0.142  0.157  0.038  0.015
dog    0.020  0.037  0.089  0.163  0.052  0.279  0.105  0.201  0.023  0.031
frog   0.013  0.057  0.061  0.064  0.106  0.008  0.570  0.058  0.025  0.038
horse  0.030  0.049  0.028  0.037  0.080  0.058  0.058  0.575  0.021  0.064
ship   0.109  0.103  0.010  0.003  0.000  0.006  0.009  0.008  0.717  0.035
truck  0.016  0.261  0.005  0.013  0.003  0.001  0.040  0.033  0.178  0.450 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.426  0.083  0.038  0.012  0.009  0.002  0.016  0.035  0.348  0.031
autom  0.016  0.676  0.004  0.002  0.001  0.003  0.023  0.015  0.164  0.096
bird   0.099  0.040  0.283  0.060  0.151  0.040  0.120  0.117  0.069  0.021
cat    0.046  0.040  0.065  0.233  0.052  0.135  0.180  0.146  0.039  0.064
deer   0.057  0.028  0.131  0.050  0.355  0.019  0.152  0.159  0.039  0.010
dog    0.029  0.033  0.092  0.151  0.062  0.282  0.097  0.196  0.041  0.017
frog   0.005  0.050  0.068  0.044  0.091  0.009  0.592  0.071  0.019  0.051
horse  0.031  0.039  0.031  0.049  0.061  0.060  0.055  0.581  0.026  0.067
ship   0.111  0.086  0.013  0.010  0.001  0.004  0.010  0.014  0.707  0.044
truck  0.022  0.277  0.007  0.006  0.003  0.003  0.045  0.034  0.210  0.393 
'''
