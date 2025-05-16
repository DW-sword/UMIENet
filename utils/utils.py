import os
import random
from collections import OrderedDict

import numpy as np
import torch
import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, model_name,epoch, loss, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, model_name + '_' + 'epoch_' + str(epoch) + '_' + str(loss) + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        # if key.startswith('module'):
        #     name = key[7:]
        # else:
        #     name = key
        # new_state_dict[name] = value
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

