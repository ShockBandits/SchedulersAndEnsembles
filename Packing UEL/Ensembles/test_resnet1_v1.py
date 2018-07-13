from Cifar10_Ensemble.Cifar10_Classifiers.ResNetV1 import *
rnv1 = ResNet_V1_Classifier("RNV1_1_test")
rnv1.get_train_data('data_batch_1')
rnv1.get_test_data('test_batch')
rnv1.create()
rnv1.fit()
rnv1.get_metrics()
rnv1.print_acc()
rnv1.print_conf_matrix('train')
rnv1.print_conf_matrix('test')
rnv1.save()

rnv11 = ResNet_V1_Classifier("RNV1_1_test")
rnv11.read()
rnv11.print_acc()
rnv11.print_conf_matrix('train')
rnv11.print_conf_matrix('test')

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_resnet1_v1.py
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
('Learning rate: ', 0.001)
Fitting RNV1_1_test
Epoch 1/1
10000/10000 [==============================] - 75s 8ms/step - loss: 2.1675 - acc: 0.2438
Using real-time data augmentation.
Epoch 1/5
313/313 [==============================] - 88s 283ms/step - loss: 1.8781 - acc: 0.3444 - val_loss: 1.7740 - val_acc: 0.3741
Epoch 2/5
313/313 [==============================] - 89s 284ms/step - loss: 1.7487 - acc: 0.3885 - val_loss: 1.6859 - val_acc: 0.4098
Epoch 3/5
313/313 [==============================] - 88s 282ms/step - loss: 1.6605 - acc: 0.4282 - val_loss: 1.5788 - val_acc: 0.4580
Epoch 4/5
313/313 [==============================] - 87s 278ms/step - loss: 1.5647 - acc: 0.4610 - val_loss: 1.4591 - val_acc: 0.4986
Epoch 5/5
313/313 [==============================] - 88s 280ms/step - loss: 1.4958 - acc: 0.4901 - val_loss: 1.4651 - val_acc: 0.4851
10000/10000 [==============================] - 12s 1ms/step
10000/10000 [==============================] - 12s 1ms/step
Training Acc: 0.5029
Testing Acc: 0.4851

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.271  0.169  0.019  0.025  0.013  0.032  0.014  0.015  0.422  0.021
autom  0.001  0.911  0.000  0.004  0.002  0.003  0.010  0.005  0.031  0.033
bird   0.058  0.031  0.314  0.074  0.071  0.191  0.138  0.030  0.083  0.011
cat    0.005  0.031  0.025  0.260  0.029  0.411  0.153  0.021  0.044  0.023
deer   0.029  0.023  0.118  0.037  0.342  0.103  0.193  0.103  0.040  0.011
dog    0.003  0.020  0.046  0.150  0.038  0.614  0.059  0.041  0.019  0.010
frog   0.005  0.039  0.047  0.078  0.038  0.069  0.685  0.003  0.025  0.012
horse  0.008  0.030  0.021  0.048  0.115  0.183  0.016  0.534  0.018  0.027
ship   0.017  0.138  0.010  0.005  0.003  0.015  0.008  0.002  0.793  0.011
truck  0.002  0.539  0.004  0.010  0.002  0.020  0.016  0.012  0.076  0.317 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.285  0.164  0.025  0.023  0.014  0.020  0.018  0.011  0.415  0.025
autom  0.000  0.902  0.002  0.001  0.003  0.009  0.006  0.004  0.039  0.034
bird   0.059  0.031  0.262  0.079  0.090  0.212  0.155  0.021  0.077  0.014
cat    0.012  0.039  0.034  0.233  0.035  0.399  0.151  0.026  0.042  0.029
deer   0.030  0.030  0.133  0.041  0.307  0.122  0.205  0.097  0.032  0.003
dog    0.003  0.020  0.055  0.148  0.038  0.611  0.049  0.039  0.026  0.011
frog   0.000  0.037  0.045  0.084  0.052  0.053  0.683  0.009  0.022  0.015
horse  0.013  0.033  0.022  0.053  0.090  0.197  0.023  0.527  0.019  0.023
ship   0.020  0.141  0.012  0.013  0.005  0.009  0.009  0.007  0.768  0.016
truck  0.002  0.574  0.003  0.010  0.003  0.013  0.016  0.014  0.092  0.273 

Saved to  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV1_1_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV1_1_test.pkl
Training Acc: 0.5029
Testing Acc: 0.4851

       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.271  0.169  0.019  0.025  0.013  0.032  0.014  0.015  0.422  0.021
autom  0.001  0.911  0.000  0.004  0.002  0.003  0.010  0.005  0.031  0.033
bird   0.058  0.031  0.314  0.074  0.071  0.191  0.138  0.030  0.083  0.011
cat    0.005  0.031  0.025  0.260  0.029  0.411  0.153  0.021  0.044  0.023
deer   0.029  0.023  0.118  0.037  0.342  0.103  0.193  0.103  0.040  0.011
dog    0.003  0.020  0.046  0.150  0.038  0.614  0.059  0.041  0.019  0.010
frog   0.005  0.039  0.047  0.078  0.038  0.069  0.685  0.003  0.025  0.012
horse  0.008  0.030  0.021  0.048  0.115  0.183  0.016  0.534  0.018  0.027
ship   0.017  0.138  0.010  0.005  0.003  0.015  0.008  0.002  0.793  0.011
truck  0.002  0.539  0.004  0.010  0.002  0.020  0.016  0.012  0.076  0.317 


       airpl  autom   bird    cat   deer    dog   frog  horse   ship  truck
airpl  0.285  0.164  0.025  0.023  0.014  0.020  0.018  0.011  0.415  0.025
autom  0.000  0.902  0.002  0.001  0.003  0.009  0.006  0.004  0.039  0.034
bird   0.059  0.031  0.262  0.079  0.090  0.212  0.155  0.021  0.077  0.014
cat    0.012  0.039  0.034  0.233  0.035  0.399  0.151  0.026  0.042  0.029
deer   0.030  0.030  0.133  0.041  0.307  0.122  0.205  0.097  0.032  0.003
dog    0.003  0.020  0.055  0.148  0.038  0.611  0.049  0.039  0.026  0.011
frog   0.000  0.037  0.045  0.084  0.052  0.053  0.683  0.009  0.022  0.015
horse  0.013  0.033  0.022  0.053  0.090  0.197  0.023  0.527  0.019  0.023
ship   0.020  0.141  0.012  0.013  0.005  0.009  0.009  0.007  0.768  0.016
truck  0.002  0.574  0.003  0.010  0.003  0.013  0.016  0.014  0.092  0.273 
'''
