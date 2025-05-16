import warnings
warnings.filterwarnings('ignore')

import os
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_testing_data
from models import *
from utils import *
from loss import *

opt = Config('/home/lvpin/Desktop/UMIENet/config.yml')

seed_everything(opt.OPTIM.SEED)

if not os.path.exists(opt.TESTING.SAVE_DIR):
    os.makedirs(opt.TESTING.SAVE_DIR, exist_ok=True)

def test():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Data Loader
    test_dir = opt.TESTING.TEST_DIR

    test_dataset = get_testing_data(test_dir, opt.MODEL.INPUT,opt.MODEL.DEPTH, {'w': opt.TESTING.PS_W, 'h': opt.TESTING.PS_H, 'ori': opt.TESTING.ORI,'with_dep':opt.TESTING.WITH_DEP})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = Model().to(device)

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model.eval()

    size = len(test_loader)

    for _, test_data in enumerate(tqdm(test_loader)):

        inp = test_data[0].contiguous().to(device)
        dep = test_data[1].contiguous().to(device)
        file_name = test_data[2][0]

        with torch.no_grad():
            res = model(inp)
            
        save_image(res, os.path.join(opt.TESTING.SAVE_DIR, file_name))


if __name__ == '__main__':
    test()
