
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from DeepLearnerClassifier import *


class SimpleCNN_Classifier(DeepLearnerClassifier):

    cfg_spec_file = 'scnn_configspec.cfg'

    def __init__(self, cfg_file):

        super(SimpleCNN_Classifier, self).__init__(cfg_file,
                                                   SimpleCNN_Classifier.cfg_spec_file)

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

        model_input_shape = model.iput_shape[1:]
        data_shape = self.train_data.shape[1:]
        if mopdel_input_shape != data_shape:
            #Assume number of channels is min element of shape tuple
            if min(data_shape) == data_shape[-1]:
                #Convert data from channels last to channels first
                self.train_data = self.train_data.transpose(0, 3, 1, 2)
                self.test_data = self.test_data.transpose(0, 3, 1, 2)
            elif min(data_shape) == data_shape[0]:
                #Convert data from channels first to channels last
                self.train_data = self.train_data.transpose(0, 2, 3, 1)
                self.test_data = self.test_data.transpose(0, 2, 3, 1)
            else:
                print("Data Input Shape: " + str(self.train_data.shape[1:]))
                print("Model Input Shape: " + str(model_input_shape))
                sys.exit()

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        self.classifier = model
