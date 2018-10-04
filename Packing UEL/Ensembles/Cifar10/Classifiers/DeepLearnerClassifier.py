from configobj import ConfigObj, ConfigObjError, flatten_errors
import cPickle as pickle
import keras
import keras.backend as K
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
from validate import Validator

from ..Accessor.readCifar10 import getCifar10, getMetaDict

_full_path = os.path.realpath(__file__)
_dir_path, _filename = os.path.split(_full_path)
results_path = os.path.join(_dir_path, 'results_dir')
_cfg_path = os.path.join(_dir_path, 'cfg_dir')


class DeepLearnerClassifier(object):

    def __init__(self, cfg_file, cfg_spec_file):

        config = self.read_cfg_file(cfg_file, cfg_spec_file)
        param_dict = config['Parameters']
        for curr_param in param_dict:
            self.__dict__[curr_param] = param_dict[curr_param]

        self.label_names = getMetaDict()['label_names']
        self.num_classes = len(self.label_names)
        self.abbr_names = [x[0:5] for x in self.label_names]
        self.summary_dict = {'name': self.name,
                             'type': self.__class__.__name__}

        self.classifier = None

        self.train_data = None
        self.train_data_file = None
        self.train_labels = None
        self.train_label_names = None
        self.has_train_data = False
        self.train_acc = None
        self.train_conf_matrix = None

        self.test_data = None
        self.test_data_file = None
        self.test_labels = None
        self.test_label_names = None
        self.has_test_data = False
        self.test_acc = None
        self.test_conf_matrix = None

        if not os.path.isdir(results_path):
            os.makedirs(results_path, 0755)

    def read_cfg_file(self, cfg_file, cfg_spec_file):
        print "Attempting to read ", cfg_file
        try:
            config = ConfigObj(os.path.join(_cfg_path, cfg_file),
                               configspec=os.path.join(_cfg_path,
                                                       cfg_spec_file),
                               file_error=True)

        except (ConfigObjError, IOError), e:
            cfg_file_path = os.path.join(_cfg_path, cfg_file)
            print "\n\nCouldn't read '%s' : %s\n\n" % (cfg_file_path, e)
            sys.exit(1)

        validator = Validator()
        cfg_results = config.validate(validator)

        if cfg_results is not True:
            for (section_list, key, _) in flatten_errors(config,
                                                         cfg_results):
                if key is not None:
                    print 'The "%s" key in the section "%s" failed validation' % \
                          (key, ', '.join(section_list))
                else:
                    print 'The following section was missing:%s ' % \
                          ', '.join(section_list)

        return config

    def read(self, suffix=''):

        filename = self.name
        if len(suffix) > 0:
            filename +=  ('_' + suffix)

        # Load classifier architecture
        saved_file = os.path.join(results_path, filename + '.json')
        with open(saved_file, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.classifier = model_from_json(loaded_model_json)

        # load weights
        saved_file = os.path.join(results_path, filename + '.h5')
        self.classifier.load_weights(saved_file)

        saved_file = os.path.join(results_path, filename + '.pkl')
        self.summary_dict = pickle.load(open(saved_file, 'rb'))
        for curr_field in self.summary_dict:
            self.__dict__[curr_field] = self.summary_dict[curr_field]
        print "Read ", saved_file

    def save(self, suffix=''):

        filename = self.name
        if len(suffix) > 0:
            filename += '_' + suffix

        # serialize classifier to JSON
        classifier_json = self.classifier.to_json()
        outfile = os.path.join(results_path, filename + '.json')
        with open(outfile, "w") as json_file:
            json_file.write(classifier_json)

        # serialize weights to HDF5
        outfile = os.path.join(results_path, filename + '.h5')
        self.classifier.save_weights(outfile)

        # save summary info in pkl file
        outfile = os.path.join(results_path, filename + '.pkl')
        pickle.dump(self.summary_dict,
                    open(outfile, 'wb'))

        # make human readable results file
        out_str = self.get_acc_str()
        out_str += self.get_conf_matrix_str("train")
        out_str += self.get_conf_matrix_str("test")
        outfile = os.path.join(results_path, filename + '.txt')
        with open(outfile, 'w') as f:
            f.write(out_str)

        print "Saved to ", outfile

    def check_channel_order(self, check_data):
        if self.classifier and self.classifier.input_shape:

            input_shape = self.classifier.input_shape[1:]
            data_shape = check_data.shape[1:]

            if input_shape != data_shape:
                # Assume number of channels is min element of shape tuple
                if min(data_shape) == data_shape[-1]:
                    # Convert data from channels last to channels first
                    check_data = check_data.transpose(0, 3, 1, 2)
                elif min(data_shape) == data_shape[0]:
                    # Convert data from channels first to channels last
                    check_data = check_data.transpose(0, 2, 3, 1)
                else:
                    print("Data Input Shape: " + str(check_data[1:]))
                    print("Model Input Shape: " + str(input_shape))
                    sys.exit()

        return check_data



    def get_train_data(self, filename):
        cifar_dict = getCifar10(filename, reshape_data=True)
        self.train_data = cifar_dict['data']
        #if K.image_data_format() == 'channels_first':
        #    self.train_data = self.train_data.transpose(0, 3, 1, 2)
        self.train_data = self.check_channel_order(self.train_data)

        self.train_data = self.train_data.astype('float32')
        self.train_data /= 255
        labels = np.asarray(cifar_dict['labels'])
        self.train_labels = keras.utils.to_categorical(labels,
                                                       self.num_classes)
        self.has_train_data = True
        self.train_data_file = filename

    def get_test_data(self, filename):
        cifar_dict = getCifar10(filename, reshape_data=True)
        self.test_data = cifar_dict['data']
        #if K.image_data_format() == 'channels_first':
        #    self.test_data = self.test_data.transpose(0, 3, 1, 2)
        self.test_data = self.check_channel_order(self.test_data)

        self.test_data = self.test_data.astype('float32')
        self.test_data /= 255
        labels = np.asarray(cifar_dict['labels'])
        self.test_labels = keras.utils.to_categorical(labels,
                                                      self.num_classes)

        self.has_test_data = True
        self.test_data_file = filename

    def fit(self):
        if not self.has_train_data:
            print "Can't fit function without training data"
            return

        print "Fitting %s" % (self.name)
        self.classifier.fit(self.train_data, self.train_labels)

        if not self.data_augmentation:
            print('Not using data augmentation.')
            self.classifier.fit(self.train_data, self.train_labels,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                validation_data=(self.test_data, self.test_labels),
                                shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(self.train_data)

            # Fit the model on the batches generated by datagen.flow().
            self.classifier.fit_generator(datagen.flow(self.train_data,
                                                       self.train_labels,
                                                       batch_size=self.batch_size),
                                          epochs=self.epochs,
                                          validation_data=(self.test_data,
                                                           self.test_labels),
                                          workers=4)

    def get_metrics(self):
        self.train_acc = self.get_acc(self.train_data,
                                      self.train_labels)
        self.train_conf_matrix = self.get_conf_matrix(self.train_data,
                                                      self.train_labels)
        self.summary_dict['train_acc'] = self.train_acc
        self.summary_dict['train_conf_matrix'] = self.train_conf_matrix

        if self.has_test_data:
            self.test_acc = self.get_acc(self.test_data,
                                         self.test_labels)
            self.test_conf_matrix = self.get_conf_matrix(self.test_data,
                                                         self.test_labels)
            self.summary_dict['test_acc'] = self.test_acc
            self.summary_dict['test_conf_matrix'] = self.test_conf_matrix
        else:
            self.summary_dict['test_acc'] = None
            self.summary_dict['test_conf_matrix'] = None

        return (self.summary_dict['train_acc'],
                self.summary_dict['train_conf_matrix'],
                self.summary_dict['test_acc'],
                self.summary_dict['test_conf_matrix'])

    def get_sample(self, dataset, samp_num):
        temp = dataset[samp_num, :, :, :]
        return temp[np.newaxis, :]

    def get_sample_label(self, dataset, samp_num):
        temp = dataset[samp_num]
        return np.argmax(temp)

    def classify(self, sample):
        return self.classifier.predict_proba(sample)[0]

    def get_conf_matrix(self, data, true_labels):
        true_labels = np.argmax(true_labels, 1)

        pred_labels = self.classifier.predict(data)
        pred_labels = np.argmax(pred_labels, 1)
        if len(true_labels) != len(pred_labels):
            print "Error, the number of true labels != number of predicted labels"
            return

        conf_matrix = confusion_matrix(true_labels, pred_labels)
        conf_matrix = conf_matrix / conf_matrix.astype(np.float).sum(axis=1)[:, np.newaxis]

        return conf_matrix

    def get_acc(self, data, true_labels):
        return self.classifier.evaluate(data, true_labels)

    def get_conf_matrix_str(self, trvate="train"):
        if trvate.lower() == "train":
            conf_matrix = self.train_conf_matrix
            out_str = "\nConfusion Matrix - Training Set"
            if self.train_label_names is None:
                label_usage = self.train_labels.sum(axis=0)
                valid_labels = [x for ctr,x in enumerate(self.abbr_names)
                                if label_usage[ctr] >0]
                self.summary_dict['train_label_names'] = valid_labels
            else:
                valid_labels = self.train_label_names
        else:
            conf_matrix = self.test_conf_matrix
            out_str = "\nConfusion Matrix - Testing Set"
            if self.test_label_names is None:
                label_usage = self.test_labels.sum(axis=0)
                valid_labels = [x for ctr,x in enumerate(self.abbr_names)
                                if label_usage[ctr] >0]
                self.summary_dict['test_label_names'] = valid_labels
            else:
                valid_labels = self.test_label_names

    
        
        if conf_matrix is not None:
            df = pd.DataFrame(conf_matrix, index=valid_labels,
                              columns=valid_labels)
            df = df.applymap("{0:.3f}".format)
            out_str += '\n' + df.to_string() + '\n'
        else:
            out_str += "Unknown"
        return out_str

    def get_acc_str(self):
        out_str = "Training Acc: "
        if self.train_acc:
            out_str += str(self.train_acc[1])
        else:
            out_str += "Unknown"

        out_str += "\nTesting Acc: "
        if self.test_acc:
            out_str += str(self.test_acc[1])
        else:
            out_str += "Unknown"
        return out_str

    def print_conf_matrix(self, trvate="train"):
        print_str = self.get_conf_matrix_str(trvate)
        print print_str

    def print_acc(self):
        print_str = self.get_acc_str()
        print print_str

