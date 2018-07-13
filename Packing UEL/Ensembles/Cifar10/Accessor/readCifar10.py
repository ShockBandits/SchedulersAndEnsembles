import cPickle
import matplotlib.pyplot as plt
import os

default_path = '/Cifar10/batches-py'
up = lambda x: os.path.dirname(x)

def getCifar10(file, reshape_data = False, path = default_path):
    data_path = up(up(up(os.path.realpath(__file__))))+default_path
    with open(os.path.join(data_path,file), 'rb') as f:
        cifar_dict = cPickle.load(f)
        if reshape_data:
            num_images = cifar_dict['data'].shape[0]
            temp = cifar_dict['data'].reshape(num_images,3,32,32)
            cifar_dict['data'] = temp.transpose([0,2,3,1])
    return cifar_dict

def getMetaDict(path = default_path):
    data_path = up(up(up(os.path.realpath(__file__))))+default_path
    with open(os.path.join(data_path,'batches.meta'), 'rb') as f:
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
