from Ensembles.Ensemble import Ensemble
x=Ensemble('Cifar10',1)
x.load_classifiers()
x.print_all_accs()
x.get_train_data()
x.assign_members_train_data()
x.test_classifiers([0,1,2,3,4])
