from Cifar10.Classifiers.SimpleCNN import *

cfg_file = 'SCNN_4_bcddh_1_200_bcddh_1_23.cfg'
train_file = 'bcddh_1_23'
test_file = 'test_batch'

scnn = SimpleCNN_Classifier(cfg_file)
scnn.get_train_data(train_file)
scnn.get_test_data(test_file)
scnn.create()
scnn.fit()
scnn.get_metrics()
scnn.print_acc()
scnn.print_conf_matrix('train')
scnn.print_conf_matrix('test')

results_suffix = '_'.join([train_file,
                           str(scnn.epochs)])

#results_suffix = ''


scnn.save(results_suffix)

scnn1 = SimpleCNN_Classifier(cfg_file)
scnn1.read(results_suffix)
scnn1.print_acc()
scnn1.print_conf_matrix('train')
scnn1.print_conf_matrix('test')

