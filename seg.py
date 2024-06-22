import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sam_trainer import train_epoch, eval_epoch
from config import MyConfig
from Data_Loader import Images_Dataset_folder
import numpy as np
from build_sam import  build_sam_vit_l
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import warnings
from segment_anything import automatic_mask_generator, SamPredictor
from utils.ostu import OtsuFastMultithreshold
from tqdm import tqdm

otsu = OtsuFastMultithreshold()



def dice(pred, gt, smooth=1e-5):
    intersection = (2 * pred * gt).sum()
    union = pred.sum() + gt.sum()
    dice_score = (intersection + smooth) / (union + smooth)
    return dice_score.item()


def seg_and_save(img, name, flag):
    out = mask_predictor.generate(img.permute(1, 2, 0).numpy())
    img[0] = img[0] * 255
    store = []
    for j in out:
        mask = j['segmentation']
        os.makedirs(os.path.join(train_data, 'seg', flag, name[:-4]), exist_ok=True)
        store.append(mask)
    store.sort(key=lambda x: x.sum(), reverse=True)
    store = np.stack(store)
    np.save(os.path.join(train_data, 'seg', flag, name[:-4], 'mask.npy'), store)


if __name__ == '__main__':



    # 禁用特定警告
    warnings.filterwarnings("ignore", category=UserWarning)
    config = MyConfig().config

    # device setting
    config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using GPU %s' % config.gpu_id)
    else:
        print('Using CPU')
    model_name = config.model_classes[0]
    config.is_dicom = False

    print(f"now model is {model_name}")
    train_data = os.path.join("../datasets", model_name, config.train_data)
    label = os.path.join("../datasets", model_name, config.label)
    test = os.path.join("../datasets", model_name, config.test_path)
    Training_Data = Images_Dataset_folder(train_data, label, test)
    Testing_Data = Images_Dataset_folder(train_data, label, test, is_train=False)
    num_train = len(Training_Data)
    indices = Training_Data.index
    print(f'seed:{config.random_seed}')
    np.random.seed(config.random_seed)
    np.random.shuffle(indices)

    fold_size = num_train // config.K
    fold_indices = [indices[i * fold_size:(i + 1) * fold_size] for i in range(config.K)]
    writer1 = SummaryWriter()

    for fold in range(config.K):
        best_model = 1e5

        train_loader = DataLoader(dataset=Training_Data, batch_size=1, num_workers=0,
                                  shuffle=True, drop_last=True
                                  )
        valid_loader = DataLoader(dataset=Testing_Data, batch_size=1, num_workers=0,
                                  shuffle=True, drop_last=True
                                  )

        # create model

        sam = build_sam_vit_l(config, 'weights/sam_l.pth').to(config.device)
        mask_predictor = automatic_mask_generator.SamAutomaticMaskGenerator(sam, output_mode='binary_mask')
        # make directory for saving weights
        if not os.path.exists(config.snap_path):
            os.mkdir(config.snap_path)

        v_loss = 0
        v_rho_s = 0
        v_rho_p = 0
        # train & validation
        for epoch in range(0, 1):
            with torch.no_grad():

                for data in tqdm(train_loader):
                    d_img_org_0 = data['d_img_org'][0]
                    d_img_org_1 = data['d_img_org'][1]
                    names = data['name']

                    for i in range(d_img_org_0.shape[0]):
                        name = names[i]
                        seg_and_save(d_img_org_0[i], name, '0')
                        seg_and_save(d_img_org_1[i], name, '1')

                for data in tqdm(valid_loader):
                    d_img_org_0 = data['d_img_org'][0]
                    d_img_org_1 = data['d_img_org'][1]
                    names = data['name']

                    for i in range(d_img_org_0.shape[0]):
                        name = names[i]
                        seg_and_save(d_img_org_0[i], name, '0')
                        seg_and_save(d_img_org_1[i], name, '1')
