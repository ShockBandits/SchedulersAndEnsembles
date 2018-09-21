import random
import sys
sys.path.append('/home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL')

from Ensembles.Ensemble import Ensemble
from strawmen import StrawMan1

# Create Ensemble
x = Ensemble('./Cifar10/',  3)
x.load_classifiers()
x.get_train_data()
x.get_test_data()

x.assign_members_train_data()
x.assign_members_test_data()

# Create strawman
strawman = StrawMan1(x, 3)


def test_it(max_new_arrivals):
    """ Tests strawman scheduler"""
    sample_list = range(100)  # Limiting test to first 100 images
    random.shuffle(sample_list)
    done = False

    while not done:
        # Calculate new arrivals to queue
        if len(sample_list) != 0:
            num_new_arrivals = random.randint(0, max_new_arrivals)
            num_new_arrivals = min(len(sample_list),
                                   num_new_arrivals)
            curr_samps = sample_list[:num_new_arrivals]
            sample_list = sample_list[num_new_arrivals:]
            strawman.newArrival(curr_samps)
            print '\nAdded to Queue', curr_samps

        else:
            num_new_arrivals = 0

        # Let strawman do single time step of processing
        results = strawman.schedule()

        print 'Processed', len(results)
        print 'Remaining Out of Queue', len(sample_list)
        print 'Remaining In Queue', len(strawman.queue)
        print 'Queue:', strawman.queue
        print results
        print "\n==================================================================\n\n"
        if len(sample_list) == 0 and len(strawman.queue) == 0:
            done = True

# Test it
test_it(5)

'''
Sample Result:


Instantiated ResNetV1 - RNV1_5.cfg
Instantiated ResNetV1 - RNV1_4.cfg
Attempting to read  SCNN_1.cfg
Instantiated SimpleCNN - SCNN_1.cfg
Attempting to read  SCNN_0.cfg
Instantiated SimpleCNN - SCNN_0.cfg
Attempting to read  SCNN_3.cfg
Instantiated SimpleCNN - SCNN_3.cfg


Loading Classifiers
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/SCNN_1.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/SCNN_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/SCNN_3.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_5.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/SchedulersAndEnsembles/Packing UEL/Ensembles/Cifar10/Classifiers/results_dir/RNV1_4.pkl


Getting Train Data
Geting data for SimpleCNN
Geting data for SimpleCNN
Geting data for SimpleCNN
Geting data for ResNetV1
Geting data for ResNetV1

Getting Test Data
Geting data for SimpleCNN
Geting data for SimpleCNN
Geting data for SimpleCNN
Geting data for ResNetV1
Geting data for ResNetV1


Added to Queue 2
Processed 2
Remaining Out of Queue 98
Remaining In Queue 0
[(93, 7, {'SCNN_1': 7, 'SCNN_0': 7, 'SCNN_3': 7}), (73, 9, {'RNV1_4': 9, 'RNV1_5': 9})]

Added to Queue 2
Processed 2
Remaining Out of Queue 96
Remaining In Queue 0
[(53, 4, {'SCNN_1': 4, 'SCNN_0': 4, 'SCNN_3': 4}), (35, 4, {'RNV1_4': 4, 'RNV1_5': 4})]

Added to Queue 3
Processed 3
Remaining Out of Queue 93
Remaining In Queue 0
[(87, 9, {'SCNN_1': 9, 'RNV1_5': 9, 'SCNN_3': 5}), (99, 8, {'RNV1_4': 8, 'SCNN_0': 8}), (68, 4, {'RNV1_4': 4, 'SCNN_0': 6, 'SCNN_3': 4})]

Added to Queue 5
Processed 3
Remaining Out of Queue 88
Remaining In Queue 2
[(15, 9, {'SCNN_1': 7, 'RNV1_5': 9, 'SCNN_3': 9}), (34, 10, {'RNV1_4': 10, 'SCNN_0': 10}), (75, 3, {'SCNN_1': 3, 'SCNN_0': 3, 'RNV1_5': 3})]

Added to Queue 4
Processed 3
Remaining Out of Queue 84
Remaining In Queue 3
[(89, 10, {'SCNN_1': 10, 'RNV1_4': 10, 'RNV1_5': 10}), (26, 5, {'SCNN_0': 5, 'SCNN_3': 5}), (9, 2, {'RNV1_5': 2, 'SCNN_3': 2, 'SCNN_0': 2})]

Added to Queue 3
Processed 3
Remaining Out of Queue 81
Remaining In Queue 3
[(42, 4, {'SCNN_1': 4, 'RNV1_4': 6, 'SCNN_3': 4}), (37, 2, {'RNV1_5': 2, 'SCNN_0': 2}), (24, 5, {'SCNN_1': 5, 'RNV1_5': 5, 'SCNN_0': 5})]

Added to Queue 4
Processed 3
Remaining Out of Queue 77
Remaining In Queue 4
[(11, 10, {'SCNN_1': 10, 'RNV1_5': 10, 'SCNN_3': 10}), (65, 3, {'RNV1_4': 3, 'SCNN_0': 3}), (21, 1, {'RNV1_4': 1, 'RNV1_5': 3, 'SCNN_3': 1})]

Added to Queue 1
Processed 3
Remaining Out of Queue 76
Remaining In Queue 2
[(54, 9, {'RNV1_4': 9, 'RNV1_5': 9, 'SCNN_3': 9}), (94, 5, {'SCNN_1': 5, 'SCNN_0': 5}), (27, 1, {'RNV1_4': 1, 'SCNN_0': 1, 'SCNN_3': 1})]

Added to Queue 4
Processed 3
Remaining Out of Queue 72
Remaining In Queue 3
[(52, 6, {'RNV1_4': 3, 'SCNN_0': 8, 'SCNN_3': 6}), (71, 4, {'SCNN_1': 4, 'RNV1_5': 4}), (84, 3, {'SCNN_1': 3, 'RNV1_5': 3, 'RNV1_4': 3})]

Added to Queue 1
Processed 3
Remaining Out of Queue 71
Remaining In Queue 1
[(29, 7, {'SCNN_0': 7, 'SCNN_3': 7, 'RNV1_5': 4}), (49, 3, {'SCNN_1': 3, 'RNV1_4': 3}), (12, 6, {'SCNN_1': 6, 'RNV1_5': 6, 'SCNN_3': 6})]

Added to Queue 3
Processed 3
Remaining Out of Queue 68
Remaining In Queue 1
[(3, 9, {'SCNN_1': 9, 'SCNN_0': 1, 'SCNN_3': 9}), (70, 3, {'RNV1_4': 3, 'RNV1_5': 3}), (64, 7, {'RNV1_4': 7, 'SCNN_1': 7, 'SCNN_3': 7})]

Added to Queue 5
Processed 3
Remaining Out of Queue 63
Remaining In Queue 3
[(97, 1, {'SCNN_1': 3, 'RNV1_5': 1, 'SCNN_0': 1}), (72, 9, {'RNV1_4': 9, 'SCNN_3': 9}), (32, 5, {'SCNN_1': 3, 'SCNN_0': 5, 'SCNN_3': 5})]

Added to Queue 2
Processed 3
Remaining Out of Queue 61
Remaining In Queue 2
[(69, 10, {'SCNN_1': 10, 'RNV1_4': 8, 'SCNN_3': 10}), (59, 4, {'SCNN_0': 4, 'RNV1_5': 4}), (43, 7, {'RNV1_4': 7, 'SCNN_0': 7, 'SCNN_1': 7})]

Added to Queue 4
Processed 3
Remaining Out of Queue 57
Remaining In Queue 3
[(38, 10, {'SCNN_1': 10, 'RNV1_5': 10, 'RNV1_4': 10}), (46, 4, {'SCNN_0': 4, 'SCNN_3': 4}), (58, 6, {'RNV1_4': 3, 'RNV1_5': 6, 'SCNN_0': 6})]

Added to Queue 0
Processed 3
Remaining Out of Queue 57
Remaining In Queue 0
[(51, 9, {'RNV1_4': 9, 'SCNN_0': 9, 'SCNN_3': 9}), (90, 9, {'SCNN_1': 9, 'RNV1_5': 1}), (82, 2, {'SCNN_1': 2, 'RNV1_5': 2, 'SCNN_0': 2})]

Added to Queue 2
Processed 2
Remaining Out of Queue 55
Remaining In Queue 0
[(6, 2, {'RNV1_4': 2, 'SCNN_0': 10, 'SCNN_3': 2}), (48, 8, {'SCNN_1': 8, 'RNV1_5': 8})]

Added to Queue 4
Processed 3
Remaining Out of Queue 51
Remaining In Queue 1
[(50, 10, {'RNV1_4': 10, 'SCNN_0': 10, 'SCNN_3': 10}), (33, 4, {'SCNN_1': 4, 'RNV1_5': 4}), (83, 8, {'RNV1_4': 8, 'SCNN_0': 8, 'SCNN_3': 8})]

Added to Queue 3
Processed 3
Remaining Out of Queue 48
Remaining In Queue 1
[(16, 6, {'SCNN_1': 6, 'RNV1_5': 6, 'SCNN_3': 6}), (14, 10, {'RNV1_4': 10, 'SCNN_0': 10}), (92, 9, {'RNV1_4': 9, 'SCNN_1': 9, 'SCNN_3': 9})]

Added to Queue 0
Processed 1
Remaining Out of Queue 48
Remaining In Queue 0
[(45, 10, {'SCNN_0': 10, 'SCNN_3': 10, 'RNV1_5': 10})]

Added to Queue 0
Processed 0
Remaining Out of Queue 48
Remaining In Queue 0
[]

Added to Queue 2
Processed 2
Remaining Out of Queue 46
Remaining In Queue 0
[(55, 9, {'RNV1_4': 9, 'RNV1_5': 9, 'SCNN_3': 9}), (79, 9, {'SCNN_1': 9, 'SCNN_0': 9})]

Added to Queue 0
Processed 0
Remaining Out of Queue 46
Remaining In Queue 0
[]

Added to Queue 4
Processed 3
Remaining Out of Queue 42
Remaining In Queue 1
[(81, 2, {'RNV1_4': 2, 'SCNN_1': 2, 'SCNN_0': 2}), (61, 6, {'RNV1_5': 6, 'SCNN_3': 6}), (98, 1, {'SCNN_1': 1, 'RNV1_4': 1, 'SCNN_3': 1})]

Added to Queue 4
Processed 3
Remaining Out of Queue 38
Remaining In Queue 2
[(28, 10, {'SCNN_0': 10, 'SCNN_3': 10, 'RNV1_5': 10}), (40, 10, {'RNV1_4': 3, 'SCNN_1': 10}), (25, 3, {'SCNN_1': 3, 'RNV1_4': 3, 'SCNN_3': 3})]

Added to Queue 5
Processed 3
Remaining Out of Queue 33
Remaining In Queue 4
[(31, 6, {'SCNN_1': 6, 'SCNN_0': 6, 'SCNN_3': 6}), (88, 9, {'RNV1_4': 9, 'RNV1_5': 9}), (86, 1, {'RNV1_4': 1, 'SCNN_0': 4, 'SCNN_3': 3})]

Added to Queue 0
Processed 3
Remaining Out of Queue 33
Remaining In Queue 1
[(62, 7, {'RNV1_4': 7, 'RNV1_5': 7, 'SCNN_3': 7}), (0, 6, {'SCNN_1': 4, 'SCNN_0': 6}), (85, 6, {'RNV1_4': 6, 'SCNN_1': 8, 'SCNN_3': 3})]

Added to Queue 2
Processed 3
Remaining Out of Queue 31
Remaining In Queue 0
[(13, 8, {'SCNN_1': 8, 'SCNN_0': 8, 'SCNN_3': 8}), (63, 4, {'RNV1_4': 4, 'RNV1_5': 4}), (77, 4, {'RNV1_4': 4, 'RNV1_5': 4, 'SCNN_3': 4})]

Added to Queue 5
Processed 3
Remaining Out of Queue 26
Remaining In Queue 2
[(41, 7, {'RNV1_5': 7, 'SCNN_3': 7, 'SCNN_0': 7}), (23, 10, {'SCNN_1': 10, 'RNV1_4': 10}), (56, 8, {'SCNN_1': 4, 'SCNN_0': 8, 'SCNN_3': 8})]

Added to Queue 4
Processed 3
Remaining Out of Queue 22
Remaining In Queue 3
[(4, 7, {'RNV1_4': 7, 'RNV1_5': 7, 'SCNN_0': 7}), (39, 6, {'SCNN_1': 6, 'SCNN_3': 6}), (20, 8, {'SCNN_1': 8, 'RNV1_5': 8, 'SCNN_3': 8})]

Added to Queue 4
Processed 3
Remaining Out of Queue 18
Remaining In Queue 4
[(91, 3, {'RNV1_4': 4, 'RNV1_5': 3, 'SCNN_3': 3}), (96, 7, {'SCNN_1': 7, 'SCNN_0': 7}), (80, 9, {'SCNN_1': 9, 'SCNN_0': 9, 'SCNN_3': 9})]

Added to Queue 2
Processed 3
Remaining Out of Queue 16
Remaining In Queue 3
[(1, 2, {'SCNN_1': 2, 'RNV1_5': 2, 'SCNN_3': 2}), (44, 1, {'RNV1_4': 1, 'SCNN_0': 1}), (95, 7, {'RNV1_4': 7, 'SCNN_1': 7, 'SCNN_3': 7})]

Added to Queue 1
Processed 3
Remaining Out of Queue 15
Remaining In Queue 1
[(5, 7, {'RNV1_4': 7, 'SCNN_1': 7, 'SCNN_3': 7}), (30, 7, {'SCNN_0': 7, 'RNV1_5': 7}), (60, 8, {'RNV1_4': 8, 'RNV1_5': 8, 'SCNN_0': 8})]

Added to Queue 2
Processed 3
Remaining Out of Queue 13
Remaining In Queue 0
[(47, 10, {'RNV1_4': 10, 'RNV1_5': 4, 'SCNN_3': 10}), (18, 9, {'SCNN_1': 9, 'SCNN_0': 9}), (78, 4, {'SCNN_1': 4, 'SCNN_0': 6, 'RNV1_5': 4})]

Added to Queue 3
Processed 3
Remaining Out of Queue 10
Remaining In Queue 0
[(10, 1, {'SCNN_1': 1, 'SCNN_0': 1, 'RNV1_4': 1}), (74, 1, {'RNV1_5': 1, 'SCNN_3': 10}), (22, 5, {'SCNN_1': 5, 'RNV1_5': 3, 'RNV1_4': 5})]

Added to Queue 5
Processed 3
Remaining Out of Queue 5
Remaining In Queue 2
[(66, 2, {'SCNN_1': 2, 'RNV1_5': 2, 'SCNN_0': 2}), (36, 5, {'RNV1_4': 5, 'SCNN_3': 5}), (76, 10, {'RNV1_4': 10, 'SCNN_0': 10, 'SCNN_3': 10})]

Added to Queue 5
Processed 3
Remaining Out of Queue 0
Remaining In Queue 4
[(2, 9, {'SCNN_0': 9, 'SCNN_3': 9, 'RNV1_5': 9}), (7, 5, {'RNV1_4': 7, 'SCNN_1': 5}), (67, 1, {'RNV1_4': 3, 'SCNN_0': 1, 'SCNN_3': 1})]

Added to Queue 0
Processed 3
Remaining Out of Queue 0
Remaining In Queue 1
[(19, 7, {'SCNN_1': 7, 'SCNN_0': 7, 'SCNN_3': 7}), (17, 8, {'RNV1_4': 8, 'RNV1_5': 8}), (8, 4, {'RNV1_4': 4, 'SCNN_1': 4, 'SCNN_3': 4})]

Added to Queue 0
Processed 1
Remaining Out of Queue 0
Remaining In Queue 0
[(57, 4, {'SCNN_1': 4, 'RNV1_5': 4, 'SCNN_3': 4})]

Process finished with exit code 0
'''