from . import *

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix

class SimpleCNN_Classifier:

    cfg_spec_file = 'scnn_configspec.cfg'

    def __init__(self, cfg_file):
        
        config = read_cfg_file(cfg_file,
                               SimpleCNN_Classifier.cfg_spec_file)
        param_dict = config['Parameters']
        for curr_param in param_dict:
            self.__dict__[curr_param] = param_dict[curr_param]
        
        self.has_train_data = False
        self.has_test_data = False
        self.train_acc = None
        self.test_acc = None
        self.train_conf_matrix = None
        self.test_conf_matrix = None        

        self.label_names = getMetaDict()['label_names']
        self.num_classes = len(self.label_names)
        self.abbr_names =[x[0:5] for x in self.label_names]
        self.summary_dict = {'name':self.name, 'type':self.__class__.__name__}

        self.classifier = None


    def create(self):
        if not self.has_train_data:
            print "Need to load training data before creating net"
            return
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.train_data.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        self.classifier = model

    def read(self):

        # Load classifier architecture
        saved_file = os.path.join(results_path, self.name + '.json')
        with open(saved_file, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.classifier = model_from_json(loaded_model_json)
        
        # load weights
        saved_file = os.path.join(results_path, self.name + '.h5')
        self.classifier.load_weights(saved_file)
        
        saved_file = os.path.join(results_path, self.name + '.pkl')
        self.summary_dict = pickle.load(open(saved_file, 'rb'))
        for curr_field in self.summary_dict:
            self.__dict__[curr_field] = self.summary_dict[curr_field]
        print "Read ", saved_file
 
    def save(self):

        # serialize classifier to JSON
        classifier_json = self.classifier.to_json()
        outfile = os.path.join(results_path, self.name + '.json')
        with open(outfile, "w") as json_file:
            json_file.write(classifier_json)
            
        # serialize weights to HDF5
        outfile = os.path.join(results_path, self.name + '.h5')
        self.classifier.save_weights(outfile)

        outfile = os.path.join(results_path, self.name + '.pkl')
        pickle.dump(self.summary_dict,
                    open(outfile, 'wb'))
        print "Saved to ",outfile

    def get_train_data(self, filename):
        cifar_dict = getCifar10(filename, reshape_data = True)
        self.train_data = cifar_dict['data']
        if K.image_data_format() == 'channels_first':
            self.train_data = self.train_data.transpose(0,3,1,2)

        self.train_data = self.train_data.astype('float32')
        self.train_data /= 255
        labels = np.asarray(cifar_dict['labels'])
        self.train_labels = keras.utils.to_categorical(labels,
                                                       self.num_classes)
        
        self.has_train_data = True
        self.train_data_file = filename
        
    def get_test_data(self, filename):
        cifar_dict = getCifar10(filename, reshape_data = True)
        self.test_data = cifar_dict['data']
        if K.image_data_format() == 'channels_first':
            self.test_data = self.test_data.transpose(0,3,1,2)
            
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
        
        print "Fitting %s"%(self.name)
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
        temp = dataset[samp_num,:,:,:]
        return temp[np.newaxis,:]
            
    def classify(self, sample):
        return  self.classifier.predict_proba(sample)[0]

    def get_conf_matrix(self):
        if not self.has_test_data:
            print "Error, No test data."
            return
        pred_labels = self.classifier.predict(self.test_data)
        pred_labels = np.argmax(pred_labels, 1)
        test_labels = np.argmax(self.test_labels, 1)
        
        if len(test_labels) != len(pred_labels):
            print "Error, the number of true labels != number of predicted labels"
            return
        conf_matrix = confusion_matrix(test_labels, pred_labels)
        conf_matrix = conf_matrix/conf_matrix.astype(np.float).sum(axis=1)
            
        return conf_matrix

    def get_acc(self, data, true_labels):
        return self.classifier.evaluate(data, true_labels)

    def print_conf_matrix(self, trvate="train"):
        if trvate.lower() == "train":
            conf_matrix = self.train_conf_matrix
            out_label = "Confusion Matrix - Training Set"
        else:
            conf_matrix = self.test_conf_matrix
            out_label = "Confusion Matrix - Testing Set"

        if conf_matrix is not None:
            df = pd.DataFrame(conf_matrix, index=self.abbr_names,
                              columns = self.abbr_names)
            df = df.applymap("{0:.3f}".format)
            print '\n',df,'\n'
        else:
            print "Unknown"

    def print_acc(self):
        print "Training Acc:", 
        if self.train_acc:
            print self.train_acc[1]
        else:
            print "Unknown"

        print "Testing Acc:", 
        if self.test_acc:
            print self.test_acc[1]
        else:
            print "Unknown"

        

        
        
