import cPickle
import matplotlib.pyplot as plt
import os
import sys
home_dir = os.path.expanduser('~')
sys.path.append(os.path.join(home_dir, "cfg_SchedulerEnsembles"))
from default_cifar_path import *

def getCifar10(file, reshape_data = False, path = default_path):
    with open(os.path.join(default_path,file), 'rb') as f:
        cifar_dict = cPickle.load(f)
        if reshape_data:
            num_images = cifar_dict['data'].shape[0]
            cifar_dict['data'] = cifar_dict['data'].reshape(num_images,3,32,32)
            #cifar_dict['data'] = cifar_dict['data'].transpose([0,2,3,1])
    return cifar_dict

def getMetaDict(path = default_path):
    with open(os.path.join(default_path, 'batches.meta'), 'rb') as f:
        label_dict = cPickle.load(f)
    return label_dict

def showImage(img, label=None):
    plt.imshow(img)
    plt.show()



'''
>>> z=getCifar10('data_batch_1')
>>> type(z)
<type 'dict'>
>>> z.keys()
['data', 'labels', 'batch_label', 'filenames']

>>> type(z['data'])
<type 'numpy.ndarray'>
>>> z['data'].shape
(10000, 3072)

>>> type(z['labels'])
<type 'list'>
>>> len(z['labels'])
10000

>>> type(z['batch_label'])
<type 'str'>
>>> z['batch_label']
'training batch 1 of 5'

>>> type(z['filenames'])
<type 'list'>
>>> len(z['filenames'])
10000


>>> zz=getMetaDict()
>>> zz
{'num_cases_per_batch': 10000, 'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_vis': 3072}
'''
