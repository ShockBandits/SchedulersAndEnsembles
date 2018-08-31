import sys
sys.path.append('/home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL')

from Ensembles.Ensemble import Ensemble
x = Ensemble('./Cifar10/',  3)
x.load_classifiers()
x.get_train_data()
x.get_test_data()

x.assign_members_train_data()
x.assign_members_test_data()

results = []
for subset_size in range(1,6):
    results.append(x.rnd_subsets_test(range(len(x.test_labels)),
                   subset_size))

for subset_size in range(1,6):
    print subset_size,results[subset_size-1]

'''
1 {'acc': 0.7461, 'wrong': 2539.0, 'right': 7461.0, 'total_time_slots': 2000}
2 {'acc': 0.8058972017831247, 'wrong': 1807.0, 'right': 7502.5, 'total_time_slots': 3334}
3 {'acc': 0.8229910634174348, 'wrong': 1644.0, 'right': 7643.666666666637, 'total_time_slots': 5000}
4 {'acc': 0.7905550444648881, 'wrong': 2049.0, 'right': 7734.0, 'total_time_slots': 5000}
5 {'acc': 0.8168759755562583, 'wrong': 1795.0, 'right': 8007.0999999999985, 'total_time_slots': 10000}

1 {'acc': 0.7507, 'wrong': 2493.0, 'right': 7507.0, 'total_time_slots': 2000}
2 {'acc': 0.8019653117113247, 'wrong': 1844.0, 'right': 7467.5, 'total_time_slots': 3334}
3 {'acc': 0.8239448008193623, 'wrong': 1633.0, 'right': 7642.499999999974, 'total_time_slots': 5000}
4 {'acc': 0.7889136162484333, 'wrong': 2063.0, 'right': 7710.25, 'total_time_slots': 5000}
5 {'acc': 0.8168759755562583, 'wrong': 1795.0, 'right': 8007.0999999999985, 'total_time_slots': 10000}

1 {'acc': 0.7469, 'wrong': 2531.0, 'right': 7469.0, 'total_time_slots': 2000}
2 {'acc': 0.8070524899057874, 'wrong': 1792.0, 'right': 7495.5, 'total_time_slots': 3334}
3 {'acc': 0.8276469530828685, 'wrong': 1598.0, 'right': 7673.666666666641, 'total_time_slots': 5000}
4 {'acc': 0.791719973389284, 'wrong': 2035.0, 'right': 7735.5, 'total_time_slots': 5000}
5 {'acc': 0.8168759755562583, 'wrong': 1795.0, 'right': 8007.0999999999985, 'total_time_slots': 10000}
'''