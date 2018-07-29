from Ensemble import Ensemble
x=Ensemble('./Cifar10/', 3)
x.load_classifiers()
x.print_all_accs()
x.get_train_data()
x.assign_members_train_data()
x.get_test_data()
x.assign_members_test_data()
z=x.get_conf_matrix()
print z

'''
/usr/bin/python2.7 "/home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/test_ensemble.py"
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
Loaded Config Info For ResNetV1 - RNV1_5.cfg
Loaded Config Info For ResNetV1 - RNV1_4.cfg
Loaded Config Info For ResNetV2 - RNV2_1.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_5.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_4.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV2_1.pkl


ResNetV1 - Classifier 0 (RNV1_5):
Training Acc: 0.9598531007478177
Testing Acc: 0.7617

ResNetV1 - Classifier 1 (RNV1_4):
Training Acc: 0.9586167394615597
Testing Acc: 0.8171


ResNetV2 - Classifier 0 (RNV2_1):
Training Acc: 0.9465
Testing Acc: 0.7905



Getting Train Data
Geting data for ResNetV1
Geting data for ResNetV1
Geting data for ResNetV2

Current Image Number: 0
Current Image Label: 6
RNV1_5  - Most Probable Label: 6
[4.5813658e-10 1.2556368e-06 1.9041836e-04 8.4171938e-03 2.2707325e-06
 3.1834941e-03 9.8820168e-01 1.1468115e-06 2.4752121e-06 8.5192234e-08] 

RNV1_4  - Most Probable Label: 6
[4.7483186e-09 1.0977631e-06 2.9439353e-03 2.8350323e-01 5.4441698e-06
 1.5711127e-02 6.9782406e-01 8.4383455e-06 2.3316411e-06 2.8655464e-07] 

RNV2_1  - Most Probable Label: 6
[1.0417554e-14 1.0838563e-09 2.4509524e-07 1.0012677e-06 7.2939891e-08
 6.7514395e-07 9.9999785e-01 1.1112377e-07 6.2226205e-12 8.1643730e-13] 

---------------------------------------------------

Current Image Number: 1
Current Image Label: 9
RNV1_5  - Most Probable Label: 9
[2.2055044e-04 6.2704799e-05 4.6108843e-07 4.1987562e-08 1.2922162e-10
 2.1104416e-12 6.4767476e-11 1.3770690e-10 3.5849520e-05 9.9968052e-01] 

RNV1_4  - Most Probable Label: 9
[8.3851104e-09 6.1985602e-06 6.6776778e-08 1.1436804e-08 6.7058103e-15
 2.3320779e-10 3.0554167e-11 3.6762446e-09 1.4323102e-07 9.9999356e-01] 

RNV2_1  - Most Probable Label: 9
[7.0709688e-11 1.0777926e-07 3.8087554e-16 2.8568733e-16 7.6688580e-20
 3.4065679e-20 5.6125859e-19 9.7224658e-14 3.1994305e-12 9.9999988e-01] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
RNV1_5  - Most Probable Label: 9
[2.3508206e-02 5.9400667e-03 1.3086294e-02 4.2900224e-03 5.2805652e-04
 4.7459820e-05 2.7725141e-04 5.7794964e-03 5.3595523e-03 9.4118357e-01] 

RNV1_4  - Most Probable Label: 9
[4.7237519e-03 2.3242849e-04 4.5385390e-05 2.9401099e-03 5.0281933e-06
 1.7249615e-05 6.1678111e-05 2.0995620e-04 5.6205234e-03 9.8614383e-01] 

RNV2_1  - Most Probable Label: 9
[2.0724286e-03 1.4094274e-04 2.3750777e-06 2.7734803e-05 7.5255485e-08
 7.0178379e-07 6.6607492e-07 4.9232753e-08 4.5861819e-04 9.9729627e-01] 

---------------------------------------------------

Current Image Number: 3
Current Image Label: 4
RNV1_5  - Most Probable Label: 4
[3.4780510e-09 6.1525851e-09 4.1886606e-06 7.3523125e-07 9.9963152e-01
 3.5821502e-06 1.8218603e-05 3.4171500e-04 2.2559747e-09 3.9340735e-09] 

RNV1_4  - Most Probable Label: 4
[1.39197725e-11 1.09691922e-10 4.69723638e-08 2.88082738e-05
 9.99813497e-01 9.81923731e-05 1.23909524e-06 5.82424509e-05
 4.46486977e-13 4.32422265e-10] 

RNV2_1  - Most Probable Label: 4
[8.7109160e-11 3.2410723e-12 2.7883509e-09 5.2765947e-09 9.9983919e-01
 2.2275874e-09 1.9353249e-05 1.4145448e-04 4.6143848e-14 1.4182132e-10] 

---------------------------------------------------

Current Image Number: 4
Current Image Label: 1
RNV1_5  - Most Probable Label: 1
[1.0751014e-10 1.0000000e+00 1.1476553e-17 2.8091049e-18 8.4919933e-19
 1.5859594e-22 1.1303479e-20 2.7445959e-19 3.6111860e-12 4.0911868e-10] 

RNV1_4  - Most Probable Label: 1
[6.0570033e-13 1.0000000e+00 7.8689201e-20 1.8276046e-19 2.8209388e-18
 2.1136218e-20 1.3353506e-19 9.1220558e-19 1.6317844e-14 2.4076745e-09] 

RNV2_1  - Most Probable Label: 1
[3.4592187e-08 9.9999988e-01 4.6785406e-18 5.2553447e-20 4.8010835e-19
 2.4194064e-22 1.9105870e-24 2.0996096e-17 1.0153685e-14 6.2677437e-08] 

---------------------------------------------------


Process finished with exit code 0
'''


'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_ensemble.py
Loaded Config Info For RandomForest - RFC_0.cfg
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
Loaded Config Info For ResNetV1 - RNV1_0.cfg
Loaded Config Info For ResNetV2 - RNV2_0.cfg
Loaded Config Info For SimpleCNN - SCNN_0.cfg
Loaded Config Info For XGBoost - XGBC_0.cfg


Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/SCNN_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RFC_0_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/XGBC_0_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV1_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV2_0.pkl


SimpleCNN - Classifier 0:
Training Acc: 0.4682
Testing Acc: 0.4602


RandomForest - Classifier 0:
Training Acc: 0.9998
Testing Acc: 0.369


XGBoost - Classifier 0:
Training Acc: 0.4969
Testing Acc: 0.3736


ResNetV1 - Classifier 0:
Training Acc: 0.536
Testing Acc: 0.5255


ResNetV2 - Classifier 0:
Training Acc: 0.4412
Testing Acc: 0.4226



Geting data for SimpleCNN
Geting data for RandomForest
Geting data for XGBoost
Geting data for ResNetV1
Geting data for ResNetV2

Current Image Number: 0
Current Image Label: 6
SCNN_0  - Most Probable Label: 6
[0.00493265 0.00538471 0.07782137 0.18491341 0.09949565 0.16369614
 0.35249028 0.10274386 0.00321356 0.00530834] 

RFC_0_test  - Most Probable Label: 6
[0.   0.   0.   0.12 0.08 0.04 0.72 0.04 0.   0.  ] 

XGBC_0_test  - Most Probable Label: 6
[0.04448942 0.08247637 0.0636211  0.11081897 0.11816238 0.13331872
 0.2502704  0.09162482 0.05363873 0.05157913] 

RNV1_0  - Most Probable Label: 6
[0.00252543 0.00194927 0.04296453 0.24760166 0.10448487 0.16670951
 0.34550485 0.08227319 0.00215056 0.00383614] 

RNV2_0  - Most Probable Label: 6
[0.00456388 0.00656152 0.06214632 0.13681659 0.18055741 0.08240572
 0.3901778  0.13086216 0.00166044 0.00424819] 

---------------------------------------------------

Current Image Number: 1
Current Image Label: 9
SCNN_0  - Most Probable Label: 1
[0.02636685 0.43615866 0.01188423 0.01771963 0.0054983  0.01005018
 0.00942942 0.01214286 0.09547377 0.37527615] 

RFC_0_test  - Most Probable Label: 9
[0.   0.04 0.   0.   0.   0.   0.   0.04 0.04 0.88] 

XGBC_0_test  - Most Probable Label: 1
[0.06757198 0.21858148 0.07379811 0.08989111 0.05528509 0.11281931
 0.04694833 0.07636517 0.13158351 0.12715591] 

RNV1_0  - Most Probable Label: 1
[6.4012213e-03 6.1779320e-01 1.0802649e-03 1.1555954e-03 1.3745519e-04
 8.9393946e-04 1.7492965e-04 1.4449517e-03 2.7169559e-02 3.4374893e-01] 

RNV2_0  - Most Probable Label: 1
[3.8466260e-02 6.2289596e-01 1.5269503e-03 2.9471584e-03 5.3817942e-04
 1.3885682e-04 3.3574572e-04 2.0044297e-03 6.8711393e-02 2.6243508e-01] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
SCNN_0  - Most Probable Label: 9
[0.08069818 0.09506809 0.04614895 0.05866797 0.01920338 0.02834463
 0.02181115 0.11550109 0.09595361 0.43860295] 

RFC_0_test  - Most Probable Label: 9
[0.   0.   0.   0.04 0.04 0.   0.   0.   0.   0.92] 

XGBC_0_test  - Most Probable Label: 9
[0.07239086 0.05411108 0.03022318 0.04112104 0.02919118 0.03293592
 0.04870701 0.07056148 0.13800466 0.48275363] 

RNV1_0  - Most Probable Label: 9
[0.17860048 0.12075048 0.00373171 0.00363354 0.00157554 0.0006626
 0.00396123 0.00307366 0.19401525 0.48999554] 

RNV2_0  - Most Probable Label: 9
[0.12597184 0.04510085 0.06534275 0.02387517 0.05191361 0.0044923
 0.03046357 0.07226986 0.1619905  0.4185796 ] 

---------------------------------------------------

Current Image Number: 3
Current Image Label: 4
SCNN_0  - Most Probable Label: 6
[0.0066687  0.00869478 0.14326897 0.07690764 0.21070375 0.0368927
 0.47997636 0.02119407 0.0086338  0.00705923] 

RFC_0_test  - Most Probable Label: 4
[0.08 0.04 0.08 0.   0.6  0.08 0.12 0.   0.   0.  ] 

XGBC_0_test  - Most Probable Label: 6
[0.04054898 0.10011289 0.07193682 0.10856446 0.13720421 0.09107311
 0.29155833 0.06235288 0.0400004  0.05664795] 

RNV1_0  - Most Probable Label: 6
[0.00315002 0.00570574 0.05364001 0.07791331 0.28172147 0.04457775
 0.44974637 0.07518174 0.00103092 0.00733263] 

RNV2_0  - Most Probable Label: 6
[0.00313707 0.00841996 0.06320436 0.10323811 0.19281484 0.04343559
 0.51601815 0.06374874 0.00270516 0.00327795] 

---------------------------------------------------

Current Image Number: 4
Current Image Label: 1
SCNN_0  - Most Probable Label: 1
[0.07197342 0.4024629  0.005332   0.00190668 0.00357094 0.00084197
 0.00214547 0.00336547 0.20668961 0.30171162] 

RFC_0_test  - Most Probable Label: 1
[0.04 0.76 0.04 0.   0.   0.   0.   0.04 0.04 0.08] 

XGBC_0_test  - Most Probable Label: 1
[0.09413977 0.25770804 0.06434504 0.05326787 0.0347997  0.04238922
 0.04687475 0.05460759 0.12250916 0.22935888] 

RNV1_0  - Most Probable Label: 1
[4.2426577e-03 6.4905697e-01 1.6993195e-05 1.8726561e-05 1.3918575e-05
 2.7321441e-06 1.2317999e-05 4.4538592e-05 6.5754694e-03 3.4001568e-01] 

RNV2_0  - Most Probable Label: 9
[4.23816517e-02 3.27405453e-01 9.06154921e-04 7.42156000e-04
 7.13584304e-04 2.56930125e-05 2.62055750e-04 3.23782256e-03
 1.15317926e-01 5.09007573e-01] 

---------------------------------------------------
'''
