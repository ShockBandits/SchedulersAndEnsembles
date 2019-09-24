Classifiers:

Currently, only working with classifiers for Cifar10. All the information for Cifar10 classifiers is in the Cifar10 sub-dir of the Ensembles dir. The Cifar10 sub-dir has 3 sub-dirs - Classifiers, Accessor and cfg_dir.

Accessor is a dir particular to Cifar10 classifiers used to access the raw data.

cfg_dir is a dir containing the configuration files for the ensemble. Each configuration file has 3 main fields:

1. classifier_root_dir  - This tells code where to look for the ensemble cfg file

   e.g. classifier_root_dir = Classifiers/cfg_dir

2. [classifiers] - This is a list of classifier types, each of which is set equal to a list of cfg files for each specific classifier

    e.g.

    [classifiers]
      RandomForest = RFC_0.cfg
      ResNetV1 = RNV1_0.cfg
      ResNetV2 = RNV2_0.cfg
      SimpleCNN = SCNN_0.cfg
      XGBoost = XGBC_0.cfg

    or

    [classifiers]
    SimpleCNN = SCNN_1.cfg, SCNN_0.cfg, SCNN_3.cfg, SCNN_4_bcdd_1_200_bcdd_1_23.cfg, SCNN_4_bcddh_1_200_bcddh_1_23.cfg

3. [datasets] - this specifies the training and testing data sets

    e.g.
    [datasets]
      train_data_file = 'data_batch_1'
      test_data_file = 'test_batch'


The Classifier dir contains the actual classifiers. Ultimately, there should've been an abstract "Classifier" class. Instead, each classifier has its own file containing a class named <classifier_type>_Classifier. Any classifier class should contain all the methods in the existing classifier classes. The only exception is for the sub-classes of DeepLearnerClassifier.

Within ths Classifiers directory, there is another cfg_dir subdirectory, containing two types of configuration files. One type is the <classifier_type>_configspec.cfg file that contains info on general parameters for classifiers of that type:

e.g.

[Parameters]
name = string(min=1, max=100, default=RFC_temp)
train_data_file = string(min=1, max=100, default=data_batch_1)
test_data_file = string(min=1, max=100, default=test_batch)

batch_size = integer(min=1, max=10000, default=32)
epochs = integer(min=1, max=10000, default=5)

data_augmentation = boolean(default=True)

and standard cfg files containing parameters for specific classifiers -

e.g.

[Parameters]
    name = 'SCNN_4'

    batch_size = 32
    epochs = 200
    data_augmentation = True


Finally, there is a dir called "results_dir" in the Classifiers dir that contains results from ecperiments.

To build ensembles, see the way the various "test_" files in the Ensembles dir interact with the Ensembles class.
