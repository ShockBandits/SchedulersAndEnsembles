from dataDisplay import Data_Display
from arbitrary_sets import *
from readCifar10 import getMetaDict

# Get meta data for Cifar 10 data set
cifar10_meta = getMetaDict()
label_dict = cifar10_meta['label_names']

# Select 3 classes from 2 data batches
default_path = '/media/innovationcommons/DataStorage/Cifar-10/cifar-10-batches-py/'
chosen_classes = [3, 5, 7] # cat, dog, horse
y = get_chosen_classes(chosen_classes, ['data_batch_1'])
yy = combine_batches(y)

# Verify that combined data batches are still labeled accurately
data = yy['data']
labels = yy['labels']
num_images = data.shape[0]
temp = data.reshape(num_images, 3, 32, 32)
data = temp.transpose([0, 2, 3, 1])
disp = Data_Display(data, labels, label_dict)
disp.start_display()

# Set an arbitrary class distribution
perc_dict = {3: 0.25, 5: 0.50, 7: 0.25}
yyy = set_class_distribution(yy, perc_dict, 'cat_dog_horse_50_25_25')

# Test resulting data set still labeled accurately
data1 = yyy['data']
labels1 = yyy['labels']
num_images1 = data1.shape[0]
temp1 = data1.reshape(num_images1, 3, 32, 32)
data1 = temp1.transpose([0, 2, 3, 1])
disp = Data_Display(data1, labels1, label_dict)
disp.start_display()

# Check class distribution is as expected
ctr_dict = defaultdict(int)
for x in labels1:
    ctr_dict[x] += 1
print ctr_dict

# Randomize image order in dataset with chosen class distribution
yy2 = shuffle_data_set(yyy)


# Verify dataset still accurately labeled
data2 = yy2['data']
labels2 = yy2['labels']
num_images2 = data2.shape[0]
temp2 = data2.reshape(num_images1, 3, 32, 32)
data2 = temp2.transpose([0, 2, 3, 1])
disp = Data_Display(data2, labels2, label_dict)
disp.start_display()

# Save dataset
cPickle.dump(yy2, open(os.path.join(default_path,
                                    yy2['name']),
                       'wb'))
