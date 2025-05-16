import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/lvpin/Desktop/UMIENet/Transfer-Learning-Library')

from pytorch_msssim import SSIM
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from data import get_training_data, get_validation_data
from models import *
from utils import *
from loss import *
from config import Config
import os

from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss

def train():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_per_process_memory_fraction(0.8,device=device)

    # Load Config
    opt = Config('/home/lvpin/Desktop/UMIENet/config.yml')
    
    # checkpoints保存路径
    if not os.path.exists(opt.TRAINING.SAVE_DIR):
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # log文件保存路径
    log_file_path = os.path.join(opt.TRAINING.SAVE_DIR,opt.TRAINING.LOG_NAME)
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            pass
    logger = get_logger(log_file_path)

    #固定随机种子
    seed_everything(opt.OPTIM.SEED)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR
    train_dataset = get_training_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, opt.MODEL.DEPTH,opt.REAL_DIR,
                    {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H,'ori': opt.TRAINING.ORI,'with_dep':opt.TRAINING.WITH_DEP})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4,
                             drop_last=True, pin_memory=True)
    val_dataset = get_validation_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, opt.MODEL.DEPTH,opt.REAL_DIR,
                {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI,'with_dep':opt.TRAINING.WITH_DEP})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True,
                            pin_memory=True)

    # Model
    model = Model().to(device)
    domain_discriminator = DomainDiscriminator(in_feature=256,hidden_size=128).to(device)


    #Pre training 预训练权重
    if opt.TRAINING.PRETRAIN:
        if os.path.exists(opt.TRAINING.WEIGHT):
            load_checkpoint(model, opt.TRAINING.WEIGHT)
            logger.info("Resuming training from checkpoint: {}".format(opt.TRAINING.WEIGHT))
        else:
            logger.info("No checkpoint found at: {}".format(opt.TRAINING.WEIGHT))

   # loss
    criterion_enhance = Loss()
    criterion_domain_adv = DomainAdversarialLoss(domain_discriminator)

    # Optimizer & Scheduler 
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, list(model.parameters())+list(domain_discriminator.parameters())), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    start_epoch = 1 
    best_epoch = 1
    best_psnr = 0
    best_ssim = 0
    best_val_loss = 1e10
    best_domain_loss = 1e10
    size = len(val_loader)

    # training
    logger.info("Training started")
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            
            inp = data[0].contiguous().to(device)
            # dep = data[1].contiguous().to(device)
            tar = data[2].contiguous().to(device)
            real = data[3].contiguous().to(device)

            res = model(inp)
            source_features = model.extract_features(inp)
            target_features = model.extract_features(real)

            enhance_loss = criterion_enhance(res, tar ,inp)
            domain_loss = criterion_domain_adv(source_features, target_features)
            total_loss = enhance_loss + domain_loss

            # 反向传播/权重更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        scheduler.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            val_total_loss = 0
            val_domain_loss = 0
            for idx, test_data in enumerate(tqdm(val_loader)):
                
                inp = test_data[0].contiguous().to(device)
                # dep = test_data[1].contiguous().to(device)
                tar = test_data[2].contiguous().to(device)
                real = test_data[3].contiguous().to(device)

                with torch.no_grad():
                    res = model(inp)
                    source_features = model.extract_features(inp)
                    target_features = model.extract_features(real)

                    enhance_loss = criterion_enhance(res, tar ,inp)
                    domain_loss = criterion_domain_adv(source_features, target_features)
                    total_loss = enhance_loss + domain_loss

                val_total_loss += total_loss.item()
                val_domain_loss += domain_loss.item()

                # psnr越大说明失真越少，生成图像质量越高
                # ssim范围[0-1],越接近1越好
                psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                ssim += structural_similarity_index_measure(res, tar, data_range=1)

            val_total_loss /= size
            val_domain_loss /= size
            psnr /= size
            ssim /= size

            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                # save model
                save_checkpoint({'state_dict': model.state_dict()}, opt.MODEL.NAME, epoch, val_total_loss, opt.TRAINING.SAVE_DIR)
                best_epoch = epoch
            if val_domain_loss < best_domain_loss:
                best_domain_loss = val_domain_loss
            if psnr > best_psnr:
                best_psnr = psnr
            if ssim > best_ssim:
                best_ssim = ssim

            logger.info("epoch: {}, total_loss:{}, domain_loss:{}, PSNR: {}, SSIM: {},  best_epoch:{}, best_loss:{}, best_domain_loss:{}, best PSNR: {}, best ssim: {}".format(epoch,val_total_loss,val_domain_loss,psnr,ssim,best_epoch,best_val_loss,best_domain_loss,best_psnr,best_ssim))
    logger.info("Training is done")


if __name__ == '__main__':
    train()
