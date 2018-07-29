from Ensembles.Ensemble import Ensemble
ENS=Ensemble("Cifar10", 1)
ENS.load_classifiers()
ENS.get_train_data(); ENS.assign_members_train_data()
ENS.get_test_data(); ENS.assign_members_test_data()
#ENS.create_classifiers()
#ENS.fit_classifiers()
ENS.test_classifiers([1,2])
trueConfMat = ENS.get_conf_matrix()
print trueConfMat

print "\n==============================================\n"

ENS=Ensemble("Cifar10", "1a")
ENS.load_classifiers()
ENS.get_train_data(); ENS.assign_members_train_data()
ENS.get_test_data(); ENS.assign_members_test_data()
#ENS.create_classifiers()
#ENS.fit_classifiers()
ENS.test_classifiers([1,2])
trueConfMat = ENS.get_conf_matrix()
print trueConfMat

print "\n==============================================\n"

ENS=Ensemble("Cifar10", "1b")
ENS.load_classifiers()
ENS.get_train_data(); ENS.assign_members_train_data()
ENS.get_test_data(); ENS.assign_members_test_data()
#ENS.create_classifiers()
#ENS.fit_classifiers()
ENS.test_classifiers([1,2])
trueConfMat = ENS.get_conf_matrix()
print trueConfMat

print "\n==============================================\n"

ENS=Ensemble("Cifar10", "1c")
ENS.load_classifiers()
ENS.get_train_data(); ENS.assign_members_train_data()
ENS.get_test_data(); ENS.assign_members_test_data()
#ENS.create_classifiers()
#ENS.fit_classifiers()
ENS.test_classifiers([1,2])
trueConfMat = ENS.get_conf_matrix()
print trueConfMat

print "\n==============================================\n"

ENS=Ensemble("Cifar10", "1d")
ENS.load_classifiers()
ENS.get_train_data(); ENS.assign_members_train_data()
ENS.get_test_data(); ENS.assign_members_test_data()
#ENS.create_classifiers()
#ENS.fit_classifiers()
ENS.test_classifiers([1,2])
trueConfMat = ENS.get_conf_matrix()
print trueConfMat

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL$ python debug1.py
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
Loaded Config Info For ResNetV2 - RNV2_1.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV2_1.pkl


Getting Train Data
Geting data for ResNetV2

Getting Test Data
Geting data for ResNetV2

Current Image Number: 1
Current Image Label: 9
RNV2_1  - Most Probable Label: 9
[7.0709688e-11 1.0777926e-07 3.8087554e-16 2.8568733e-16 7.6688580e-20
 3.4065679e-20 5.6125859e-19 9.7224658e-14 3.1994305e-12 9.9999988e-01] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
RNV2_1  - Most Probable Label: 9
[2.0724286e-03 1.4094274e-04 2.3750777e-06 2.7734803e-05 7.5255485e-08
 7.0178379e-07 6.6607492e-07 4.9232753e-08 4.5861819e-04 9.9729627e-01] 

---------------------------------------------------

Geting conf_matrix
Geting conf_matrix for ResNetV2-0

==============================================

Loaded Config Info For ResNetV1 - RNV1_2.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_2.pkl


Getting Train Data
Geting data for ResNetV1

Getting Test Data
Geting data for ResNetV1

Current Image Number: 1
Current Image Label: 9
RNV1_2  - Most Probable Label: 9
[5.0225199e-11 4.7215448e-10 2.7419347e-10 1.1557596e-12 1.7475122e-15
 1.0531728e-15 6.2972326e-16 2.7199276e-10 4.6217790e-17 1.0000000e+00] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
RNV1_2  - Most Probable Label: 9
[5.1102567e-02 9.8474025e-05 1.1096478e-06 2.8884489e-05 3.8706344e-07
 7.9571244e-08 6.5540471e-06 2.2963020e-03 4.3794016e-06 9.4646126e-01] 

---------------------------------------------------

Geting conf_matrix
Geting conf_matrix for ResNetV1-0

==============================================

Loaded Config Info For ResNetV1 - RNV1_3.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_3.pkl


Getting Train Data
Geting data for ResNetV1

Getting Test Data
Geting data for ResNetV1

Current Image Number: 1
Current Image Label: 9
RNV1_3  - Most Probable Label: 9
[1.0796174e-15 1.3069399e-08 3.0019055e-22 3.9608832e-18 8.7395626e-26
 1.4656932e-22 5.5112454e-23 2.9144497e-19 9.4431383e-16 1.0000000e+00] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
RNV1_3  - Most Probable Label: 9
[8.17913329e-04 9.28086884e-05 1.76712827e-04 1.14003335e-04
 2.86243509e-08 2.66126472e-06 9.99669325e-08 4.22249468e-05
 8.14321102e-04 9.97939289e-01] 

---------------------------------------------------

Geting conf_matrix
Geting conf_matrix for ResNetV1-0

==============================================

Loaded Config Info For ResNetV1 - RNV1_4.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_4.pkl


Getting Train Data
Geting data for ResNetV1

Getting Test Data
Geting data for ResNetV1

Current Image Number: 1
Current Image Label: 9
RNV1_4  - Most Probable Label: 9
[8.3851104e-09 6.1985602e-06 6.6776778e-08 1.1436804e-08 6.7058103e-15
 2.3320779e-10 3.0554167e-11 3.6762446e-09 1.4323102e-07 9.9999356e-01] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
RNV1_4  - Most Probable Label: 9
[4.7237519e-03 2.3242849e-04 4.5385390e-05 2.9401099e-03 5.0281933e-06
 1.7249615e-05 6.1678111e-05 2.0995620e-04 5.6205234e-03 9.8614383e-01] 

---------------------------------------------------

Geting conf_matrix
Geting conf_matrix for ResNetV1-0

==============================================

Loaded Config Info For ResNetV1 - RNV1_5.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SoumyaRepo/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_5.pkl


Getting Train Data
Geting data for ResNetV1

Getting Test Data
Geting data for ResNetV1

Current Image Number: 1
Current Image Label: 9
RNV1_5  - Most Probable Label: 9
[2.2055044e-04 6.2704799e-05 4.6108843e-07 4.1987562e-08 1.2922162e-10
 2.1104416e-12 6.4767476e-11 1.3770690e-10 3.5849520e-05 9.9968052e-01] 

---------------------------------------------------

Current Image Number: 2
Current Image Label: 9
RNV1_5  - Most Probable Label: 9
[2.3508206e-02 5.9400667e-03 1.3086294e-02 4.2900224e-03 5.2805652e-04
 4.7459820e-05 2.7725141e-04 5.7794964e-03 5.3595523e-03 9.4118357e-01] 

---------------------------------------------------

Geting conf_matrix
Geting conf_matrix for ResNetV1-0
'''
