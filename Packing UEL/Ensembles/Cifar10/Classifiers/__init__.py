import cPickle as pickle
from configobj import ConfigObj, ConfigObjError, flatten_errors
import numpy as np
import os
import pandas as pd
import sys
from validate import Validator

from ..Accessor.readCifar10 import getCifar10, getMetaDict

_full_path = os.path.realpath(__file__)
_dir_path, _filename = os.path.split(_full_path)
results_path = os.path.join(_dir_path, 'results_dir')
_cfg_path = os.path.join(_dir_path, 'cfg_dir')

if not os.path.isdir(results_path):
    os.makedirs(results_path, 0755)

def read_cfg_file(cfg_file, cfg_spec_file):

    try:
        config = ConfigObj(os.path.join(_cfg_path, cfg_file),
                           configspec=os.path.join(_cfg_path,
                                                   cfg_spec_file),
                           file_error=True)
        
    except (ConfigObjError, IOError), e:
        cfg_file_path = os.path.join(_cfg_path, cfg_file)
        print "\n\nCouldn't read '%s' : %s\n\n"%(cfg_file_path, e)
        sys.exit(1)
        
    validator = Validator()
    cfg_results = config.validate(validator)

    if cfg_results != True:
        for (section_list, key, _) in flatten_errors(config,
                                                     cfg_results):
            if key is not None:
                print 'The "%s" key in the section "%s" failed validation' % \
                      (key, ', '.join(section_list))
            else:
                print 'The following section was missing:%s ' % \
                      ', '.join(section_list)

    return config
