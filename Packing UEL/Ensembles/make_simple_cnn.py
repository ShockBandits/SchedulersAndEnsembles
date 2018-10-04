import argparse
from configobj import ConfigObj, ConfigObjError, flatten_errors
from Cifar10.Classifiers.SimpleCNN import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=str)
    parser.add_argument("train_file", type=str)
    parser.add_argument("test_file", type=str)
    parser.parse_args()
    
    args = parser.parse_args()

    scnn = SimpleCNN_Classifier(args.cfg_file)
    scnn.get_train_data(args.train_file)
    scnn.get_test_data(args.test_file)
    scnn.create()
    scnn.fit()
    scnn.get_metrics()
    scnn.print_acc()
    scnn.print_conf_matrix('train')
    scnn.print_conf_matrix('test')

    results_suffix = '_'.join([args.train_file,
                               str(scnn.epochs)])

    scnn.save()#results_suffix)

    scnn1 = SimpleCNN_Classifier(args.cfg_file)
    scnn1.read()#results_suffix)
    scnn1.print_acc()
    scnn1.print_conf_matrix('train')
    scnn1.print_conf_matrix('test')

