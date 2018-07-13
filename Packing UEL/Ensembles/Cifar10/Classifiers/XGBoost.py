from . import *
#from . import read_cfg_file, results_path
#from ..Cifar10_Accessor.readCifar10 import getCifar10, getMetaDict
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

class XGBoost_Classifier:

    cfg_spec_file = 'xgbc_configspec.cfg'

    def __init__(self, cfg_file):

        config = read_cfg_file(cfg_file,
                               XGBoost_Classifier.cfg_spec_file)
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
        self.classifier = XGBClassifier(n_estimators =
                                        self.n_estimators)
        self.summary_dict['classifier'] = self.classifier

    def read(self):
        saved_file = os.path.join(results_path, self.name + '.pkl')
        self.summary_dict = pickle.load(open(saved_file, 'rb'))
        for curr_field in self.summary_dict:
            self.__dict__[curr_field] = self.summary_dict[curr_field]
        print "Read ", saved_file
 
    def save(self):
        outfile = os.path.join(results_path, self.name + '.pkl')
        pickle.dump(self.summary_dict,
                    open(outfile, 'wb'))
        print "Saved to ",outfile

    def get_train_data(self, filename):
        cifar_dict = getCifar10(filename)
        self.train_data = cifar_dict['data']
        self.train_labels = np.asarray(cifar_dict['labels'])
        self.has_train_data = True
        self.train_data_file = filename
        
    def get_test_data(self, filename):
        cifar_dict = getCifar10(filename)
        self.test_data = cifar_dict['data']
        self.test_labels = np.asarray(cifar_dict['labels'])
        self.has_test_data = True
        self.test_data_file = filename
        
    def fit(self):
        if not self.has_train_data:
            print "Can't fit function without training data"
            return
        
        print "Fitting %s"%(self.name)
        self.classifier.fit(self.train_data, self.train_labels)
        self.summary_dict['classifier'] = self.classifier        

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
        temp = dataset[samp_num,:]
        return temp[np.newaxis,:]                           

    def classify(self, sample):
        return  self.classifier.predict_proba(sample)[0]

    def get_conf_matrix(self):
        if not self.has_test_data:
            print "Error, No test data."
            return
        pred_labels = self.classifier.predict(self.test_data)
        if len(self.test_labels) != len(pred_labels):
            print "Error, the number of true labels != number of predicted labels"
            return
        conf_matrix = confusion_matrix(self.test_labels, pred_labels)
        conf_matrix = conf_matrix/conf_matrix.astype(np.float).sum(axis=1)
            
        return conf_matrix

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

    def get_acc(self, data, true_labels):
        return self.classifier.score(data, true_labels)

    def print_acc(self):
        print "Training Acc:", 
        if self.train_acc:
            print self.train_acc
        else:
            print "Unknown"

        print "Testing Acc:", 
        if self.test_acc:
            print self.test_acc
        else:
            print "Unknown"

        

        
        
