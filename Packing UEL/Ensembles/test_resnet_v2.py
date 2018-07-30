from Cifar10.Classifiers.ResNetV2 import *
rnv2 = ResNetV2_Classifier("RNV2_0.cfg")
rnv2.get_train_data('data_batch_1')
rnv2.get_test_data('test_batch')
rnv2.create()
rnv2.fit()
rnv2.get_metrics()
rnv2.print_acc()
rnv2.print_conf_matrix('train')
rnv2.print_conf_matrix('test')
rnv2.save()

rnv21 = ResNetV2_Classifier("RNV2_0.cfg")
rnv21.read()
rnv21.print_acc()
rnv21.print_conf_matrix('train')
rnv21.print_conf_matrix('test')

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_resnet_v2.py 
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
('Learning rate: ', 0.001)
Fitting RNV2_0
Epoch 1/1
10000/10000 [==============================] - 250s 25ms/step - loss: 2.3947 - acc: 0.2539
Using real-time data augmentation.
Epoch 1/5
313/313 [==============================] - 247s 791ms/step - loss: 2.0280 - acc: 0.3470 - val_loss: 1.9456 - val_acc: 0.3846
Epoch 2/5
313/313 [==============================] - 248s 793ms/step - loss: 1.8661 - acc: 0.4030 - val_loss: 1.9278 - val_acc: 0.3966
Epoch 3/5
313/313 [==============================] - 248s 791ms/step - loss: 1.7743 - acc: 0.4281 - val_loss: 1.6922 - val_acc: 0.4643
Epoch 4/5
313/313 [==============================] - 248s 793ms/step - loss: 1.6715 - acc: 0.4587 - val_loss: 1.6316 - val_acc: 0.4823
Epoch 5/5
313/313 [==============================] - 248s 793ms/step - loss: 1.6092 - acc: 0.4825 - val_loss: 1.6207 - val_acc: 0.4738
10000/10000 [==============================] - 5s 494us/step
10000/10000 [==============================] - 5s 494us/step
Training Acc: 0.485
Testing Acc: 0.4738

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.502  0.007  0.209  0.016  0.023  0.004  0.006  0.025  0.132  0.076
autom  0.041  0.246  0.046  0.018  0.037  0.000  0.028  0.020  0.073  0.491
bird   0.037  0.001  0.529  0.054  0.223  0.017  0.075  0.047  0.008  0.009
cat    0.007  0.001  0.174  0.319  0.162  0.121  0.125  0.064  0.006  0.021
deer   0.004  0.000  0.223  0.033  0.518  0.010  0.077  0.117  0.004  0.014
dog    0.001  0.002  0.170  0.234  0.161  0.239  0.062  0.121  0.002  0.009
frog   0.004  0.001  0.097  0.051  0.203  0.007  0.621  0.010  0.002  0.004
horse  0.004  0.000  0.072  0.048  0.198  0.027  0.014  0.603  0.005  0.029
ship   0.236  0.008  0.115  0.018  0.020  0.002  0.016  0.008  0.510  0.067
truck  0.042  0.016  0.051  0.016  0.033  0.000  0.017  0.056  0.028  0.741 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.488  0.008  0.222  0.012  0.028  0.001  0.014  0.028  0.120  0.079
autom  0.060  0.256  0.063  0.016  0.028  0.000  0.022  0.030  0.059  0.466
bird   0.034  0.000  0.494  0.061  0.232  0.027  0.078  0.056  0.013  0.005
cat    0.012  0.002  0.186  0.313  0.169  0.105  0.131  0.065  0.006  0.011
deer   0.014  0.001  0.213  0.049  0.508  0.016  0.080  0.105  0.006  0.008
dog    0.004  0.001  0.175  0.229  0.156  0.251  0.060  0.112  0.004  0.008
frog   0.001  0.001  0.089  0.054  0.209  0.004  0.629  0.007  0.002  0.004
horse  0.011  0.000  0.070  0.044  0.189  0.041  0.024  0.589  0.000  0.032
ship   0.232  0.013  0.123  0.021  0.022  0.001  0.012  0.013  0.479  0.084
truck  0.047  0.018  0.045  0.023  0.030  0.000  0.018  0.056  0.032  0.731 

Saved to  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV2_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV2_0.pkl
Training Acc: 0.485
Testing Acc: 0.4738

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.502  0.007  0.209  0.016  0.023  0.004  0.006  0.025  0.132  0.076
autom  0.041  0.246  0.046  0.018  0.037  0.000  0.028  0.020  0.073  0.491
bird   0.037  0.001  0.529  0.054  0.223  0.017  0.075  0.047  0.008  0.009
cat    0.007  0.001  0.174  0.319  0.162  0.121  0.125  0.064  0.006  0.021
deer   0.004  0.000  0.223  0.033  0.518  0.010  0.077  0.117  0.004  0.014
dog    0.001  0.002  0.170  0.234  0.161  0.239  0.062  0.121  0.002  0.009
frog   0.004  0.001  0.097  0.051  0.203  0.007  0.621  0.010  0.002  0.004
horse  0.004  0.000  0.072  0.048  0.198  0.027  0.014  0.603  0.005  0.029
ship   0.236  0.008  0.115  0.018  0.020  0.002  0.016  0.008  0.510  0.067
truck  0.042  0.016  0.051  0.016  0.033  0.000  0.017  0.056  0.028  0.741 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.488  0.008  0.222  0.012  0.028  0.001  0.014  0.028  0.120  0.079
autom  0.060  0.256  0.063  0.016  0.028  0.000  0.022  0.030  0.059  0.466
bird   0.034  0.000  0.494  0.061  0.232  0.027  0.078  0.056  0.013  0.005
cat    0.012  0.002  0.186  0.313  0.169  0.105  0.131  0.065  0.006  0.011
deer   0.014  0.001  0.213  0.049  0.508  0.016  0.080  0.105  0.006  0.008
dog    0.004  0.001  0.175  0.229  0.156  0.251  0.060  0.112  0.004  0.008
frog   0.001  0.001  0.089  0.054  0.209  0.004  0.629  0.007  0.002  0.004
horse  0.011  0.000  0.070  0.044  0.189  0.041  0.024  0.589  0.000  0.032
ship   0.232  0.013  0.123  0.021  0.022  0.001  0.012  0.013  0.479  0.084
truck  0.047  0.018  0.045  0.023  0.030  0.000  0.018  0.056  0.032  0.731 
'''
