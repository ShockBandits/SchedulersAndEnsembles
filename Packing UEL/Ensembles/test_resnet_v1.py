from Cifar10_Ensemble.Cifar10_Classifiers.ResNetV1 import *
rnv1 = ResNetV1_Classifier("RNV1_0.cfg")
rnv1.get_train_data('data_batch_1')
rnv1.get_test_data('test_batch')
rnv1.create()
rnv1.fit()
rnv1.get_metrics()
rnv1.print_acc()
rnv1.print_conf_matrix('train')
rnv1.print_conf_matrix('test')
rnv1.save()

rnv11 = ResNetV1_Classifier("RNV1_0.cfg")
rnv11.read()
rnv11.print_acc()
rnv11.print_conf_matrix('train')
rnv11.print_conf_matrix('test')

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_resnet_v1.py 
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
('Learning rate: ', 0.001)
Fitting RNV1_0
Epoch 1/1
10000/10000 [==============================] - 75s 8ms/step - loss: 2.1990 - acc: 0.2237
Using real-time data augmentation.
Epoch 1/5
313/313 [==============================] - 87s 278ms/step - loss: 1.8480 - acc: 0.3466 - val_loss: 1.8155 - val_acc: 0.3852
Epoch 2/5
313/313 [==============================] - 87s 277ms/step - loss: 1.7237 - acc: 0.3992 - val_loss: 1.6869 - val_acc: 0.4044
Epoch 3/5
313/313 [==============================] - 87s 277ms/step - loss: 1.6215 - acc: 0.4381 - val_loss: 1.6723 - val_acc: 0.4310
Epoch 4/5
313/313 [==============================] - 87s 277ms/step - loss: 1.5398 - acc: 0.4683 - val_loss: 1.5618 - val_acc: 0.4651
Epoch 5/5
313/313 [==============================] - 87s 278ms/step - loss: 1.4602 - acc: 0.5018 - val_loss: 1.6035 - val_acc: 0.4618
10000/10000 [==============================] - 12s 1ms/step
10000/10000 [==============================] - 12s 1ms/step
Training Acc: 0.4847
Testing Acc: 0.4618

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.377  0.026  0.027  0.002  0.006  0.000  0.003  0.009  0.542  0.008
autom  0.018  0.711  0.001  0.000  0.000  0.000  0.008  0.000  0.238  0.023
bird   0.126  0.007  0.365  0.009  0.100  0.016  0.081  0.063  0.227  0.006
cat    0.026  0.014  0.152  0.134  0.066  0.110  0.144  0.103  0.225  0.027
deer   0.081  0.008  0.177  0.008  0.352  0.002  0.088  0.195  0.085  0.003
dog    0.030  0.020  0.128  0.114  0.097  0.227  0.053  0.179  0.146  0.004
frog   0.011  0.017  0.086  0.019  0.066  0.001  0.677  0.023  0.087  0.012
horse  0.048  0.020  0.033  0.013  0.085  0.025  0.018  0.652  0.082  0.024
ship   0.039  0.011  0.003  0.000  0.000  0.001  0.002  0.002  0.940  0.003
truck  0.048  0.298  0.003  0.000  0.003  0.001  0.005  0.014  0.236  0.391 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.354  0.019  0.029  0.001  0.007  0.000  0.010  0.010  0.561  0.009
autom  0.014  0.673  0.000  0.000  0.004  0.000  0.007  0.003  0.271  0.028
bird   0.104  0.010  0.360  0.011  0.111  0.025  0.096  0.063  0.213  0.007
cat    0.048  0.024  0.138  0.132  0.065  0.093  0.139  0.102  0.240  0.019
deer   0.073  0.014  0.192  0.014  0.322  0.011  0.105  0.194  0.070  0.005
dog    0.040  0.017  0.146  0.086  0.094  0.225  0.074  0.170  0.141  0.007
frog   0.004  0.012  0.107  0.019  0.075  0.002  0.646  0.024  0.102  0.009
horse  0.057  0.017  0.049  0.013  0.048  0.024  0.030  0.648  0.086  0.028
ship   0.035  0.014  0.011  0.003  0.001  0.001  0.003  0.008  0.921  0.003
truck  0.031  0.296  0.004  0.002  0.000  0.000  0.007  0.015  0.308  0.337 

Saved to  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV1_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV1_0.pkl
Training Acc: 0.4847
Testing Acc: 0.4618

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.377  0.026  0.027  0.002  0.006  0.000  0.003  0.009  0.542  0.008
autom  0.018  0.711  0.001  0.000  0.000  0.000  0.008  0.000  0.238  0.023
bird   0.126  0.007  0.365  0.009  0.100  0.016  0.081  0.063  0.227  0.006
cat    0.026  0.014  0.152  0.134  0.066  0.110  0.144  0.103  0.225  0.027
deer   0.081  0.008  0.177  0.008  0.352  0.002  0.088  0.195  0.085  0.003
dog    0.030  0.020  0.128  0.114  0.097  0.227  0.053  0.179  0.146  0.004
frog   0.011  0.017  0.086  0.019  0.066  0.001  0.677  0.023  0.087  0.012
horse  0.048  0.020  0.033  0.013  0.085  0.025  0.018  0.652  0.082  0.024
ship   0.039  0.011  0.003  0.000  0.000  0.001  0.002  0.002  0.940  0.003
truck  0.048  0.298  0.003  0.000  0.003  0.001  0.005  0.014  0.236  0.391 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.354  0.019  0.029  0.001  0.007  0.000  0.010  0.010  0.561  0.009
autom  0.014  0.673  0.000  0.000  0.004  0.000  0.007  0.003  0.271  0.028
bird   0.104  0.010  0.360  0.011  0.111  0.025  0.096  0.063  0.213  0.007
cat    0.048  0.024  0.138  0.132  0.065  0.093  0.139  0.102  0.240  0.019
deer   0.073  0.014  0.192  0.014  0.322  0.011  0.105  0.194  0.070  0.005
dog    0.040  0.017  0.146  0.086  0.094  0.225  0.074  0.170  0.141  0.007
frog   0.004  0.012  0.107  0.019  0.075  0.002  0.646  0.024  0.102  0.009
horse  0.057  0.017  0.049  0.013  0.048  0.024  0.030  0.648  0.086  0.028
ship   0.035  0.014  0.011  0.003  0.001  0.001  0.003  0.008  0.921  0.003
truck  0.031  0.296  0.004  0.002  0.000  0.000  0.007  0.015  0.308  0.337 
'''
