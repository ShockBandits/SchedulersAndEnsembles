from Cifar10_Ensemble.Ensemble import Ensemble
x=Ensemble('./Cifar10_Ensemble/cfg_dir/ensemble_0.cfg')
x.load_classifiers()
x.print_all_accs()
x.get_train_data()
x.assign_members_train_data()
x.test_classifiers([0,1,2,3,4])

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
