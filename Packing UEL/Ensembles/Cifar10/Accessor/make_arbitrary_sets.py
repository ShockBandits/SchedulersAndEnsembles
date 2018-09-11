from arbitrary_sets import *

default_path = '/media/innovationcommons/DataStorage/Cifar-10/cifar-10-batches-py/'


def make_datasets(chosen_classes, batch_list,
                  name, perc_dict=None):

    # Select 3 classes from 2 data batches
    y = get_chosen_classes(chosen_classes, batch_list)
    yy = combine_batches(y)

    # Set an arbitrary class distribution
    if perc_dict is not None:
        yy2 = set_class_distribution(yy, perc_dict, name)

        # Randomize image order in dataset with chosen class distribution
        yy = shuffle_data_set(yy2)
    else:
        yy['name'] = name

    # Save dataset
    cPickle.dump(yy, open(os.path.join(default_path, yy['name']), 'wb'))

'''
make_datasets([3, 5, 7], ['data_batch_1', 'data_batch_2'],
              'cat_dog_horse_25_50_25_db12',
              {3: 0.25, 5: 0.50, 7: 0.25})

make_datasets([3, 5, 7], ['data_batch_2', 'data_batch_3'],
              'cat_dog_horse_50_25_25_db23',
              {3: 0.50, 5: 0.25, 7: 0.25})

make_datasets([3, 5, 7], ['data_batch_3', 'data_batch_4'],
              'cat_dog_horse_25_25_50_db34',
              {3: 0.50, 5: 0.25, 7: 0.25})

make_datasets([3, 5, 7], ['data_batch_4', 'data_batch_5'],
              'cat_dog_horse_333_334_333_db45',
              {3: 0.333, 5: 0.334, 7: 0.333})

make_datasets([3, 5, 7], ['data_batch_5', 'data_batch_1'],
              'cat_dog_horse_40_40_20_db51',
              {3: 0.40, 5: 0.40, 7: 0.20})

#make_datasets([3, 5, 7], ['test_batch'],
#              'cat_dog_horse_test')

make_datasets([3, 5, 7], ['data_batch_5', 'data_batch_1'],
              'cat_dog_horse_475_475_05_db51',
              {3: 0.475, 5: 0.475, 7: 0.05})

make_datasets([1, 2, 3, 5, 7], ['test_batch'],
              'auto_bird_cat_dog_horse_test')

make_datasets([1, 2, 3, 5, 7], ['data_batch_1', 'data_batch_2'],
              'auto_bird_cat_dog_horse_20_20_20_20_20_db12',
              {1: 0.20, 2: 0.20, 3: 0.20, 5: 0.20, 7: 0.20})

make_datasets([1, 2, 3, 5, 7], ['data_batch_1', 'data_batch_2', 'data_batch_3'],
              'auto_bird_cat_dog_horse_20_20_20_20_20_db123',
              {1: 0.20, 2: 0.20, 3: 0.20, 5: 0.20, 7: 0.20})

make_datasets([1, 2, 3, 5, 7], ['data_batch_2', 'data_batch_3', 'data_batch_4'],
              'auto_bird_cat_dog_horse_15_40_15_15_15_db234',
              {1: 0.15, 2: 0.40, 3: 0.15, 5: 0.15, 7: 0.15})

make_datasets([1, 2, 3, 5, 7], ['data_batch_3', 'data_batch_4', 'data_batch_5'],
              'auto_bird_cat_dog_horse_15_15_40_15_15_db345',
              {1: 0.15, 2: 0.15, 3: 0.40, 5: 0.15, 7: 0.15})

make_datasets([1, 2, 3, 5, 7], ['data_batch_4', 'data_batch_5', 'data_batch_1'],
              'auto_bird_cat_dog_horse_15_15_15_40_15_db451',
              {1: 0.15, 2: 0.15, 3: 0.15, 5: 0.40, 7: 0.15})

make_datasets([1, 2, 3, 5, 7], ['data_batch_5', 'data_batch_1', 'data_batch_2'],
              'auto_bird_cat_dog_horse_15_15_15_15_40_db512',
              {1: 0.15, 2: 0.15, 3: 0.15, 5: 0.15, 7: 0.40})
'''

make_datasets([1, 2, 3, 5, 7], ['data_batch_5', 'data_batch_1', 'data_batch_2'],
              'auto_bird_cat_dog_horse_005_25_34_255_15_db512',
              {1: 0.005, 2: 0.25, 3: 0.34, 5: 0.255, 7: 0.15})