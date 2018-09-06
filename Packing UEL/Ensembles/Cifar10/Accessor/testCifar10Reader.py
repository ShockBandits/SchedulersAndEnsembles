from readCifar10 import getCifar10, getMetaDict
from dataDisplay import Data_Display


cifar10_info = getCifar10('data_batch_1', True)
data = cifar10_info['data']
labels = cifar10_info['labels']

cifar10_meta = getMetaDict()
label_dict = cifar10_meta['label_names']

disp = Data_Display(data, labels, label_dict)
disp.start_display()
