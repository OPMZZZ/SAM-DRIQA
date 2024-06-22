import os
from model.coatnet import get_my_coatnet
import torch
from scipy.stats import spearmanr, pearsonr
import numpy as np
from config import MyConfig
from Data_Loader import Images_Dataset_folder, chest_dataset
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
config = MyConfig().config

model_name = config.model_classes[config.now_class]

print(f"now model is {model_name}")
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')

model_name = config.model_classes[config.now_class]
models = [x for x in os.listdir(os.path.join("weights/", model_name)) if x.startswith("aba1_")]
# models = ["sam_d_ankle_K1_best_model.pth"]

for check in models:
    checkpoint = os.path.join("weights/", model_name, check)

    # create model
    net = get_my_coatnet().to(config.device)
    # load weights
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['net'])
    net.eval()


    def calc_loss(prediction, target):
        """Calculating the loss and metrics
        Args:
            prediction = predicted image
            target = Targeted image
            metrics = Metrics printed
            bce_weight = 0.5 (default)
        Output:
            loss : dice loss of the epoch """

        criteon = torch.nn.L1Loss()
        sum_loss = 0
        sample_loss = 0
        mape_loss = 0
        for i in range(target.shape[0]):
            abs_loss = criteon(prediction[i], target[i])*target.shape[1] / target[i].sum()
            sample_loss += abs_loss.item()
        total_loss = criteon(prediction, target)
        for i in range(target.shape[0]):
            p = prediction[i].sum()
            l = target[i].sum()
            sum_loss += criteon(p, l)
            mape_loss += torch.abs(p - l) / l
        return total_loss, sample_loss / target.shape[0], sum_loss / target.shape[0], mape_loss / target.shape[0]




    train_data = os.path.join("../datasets", model_name, config.train_data)
    label = os.path.join("../datasets", model_name, config.label)
    test = os.path.join("../datasets", model_name, config.test_path)

    Testing_Data = chest_dataset(train_data, label, is_train=False)
    test_loader = torch.utils.data.DataLoader(dataset=Testing_Data, batch_size=1,
                                              num_workers=0,
                                              )
    criterion = torch.nn.L1Loss()
    losses = []
    sum_losses = []
    error_data = {}
    sample_losses = []
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    mean_plcc = []
    mean_srocc = []
    mape = []
    res = {}
    for index, data in enumerate(test_loader):
        with torch.no_grad():
            d_img_org_0 = data['d_img_org'][0].to(config.device)
            d_img_org_1 = data['d_img_org'][1].to(config.device)
            # pred = net(d_img_org_0, d_img_org_1)

            pred_tag, pred, _ = net(d_img_org_0, d_img_org_1)
            tmp_value, tmp_scores = torch.topk(pred_tag, 3, dim=2)
            tmp_value = torch.softmax(tmp_value, dim=2)
            pred_tag = torch.sum(tmp_scores * tmp_value, dim=2)
            labels_tag = data['score_tag'].to(config.device)
            labels = data['score'].to(config.device)
            pred = pred
            _, _, sum_loss, mape_loss = calc_loss(pred, labels)
            sum_losses.append(sum_loss.item())
            mape.append(mape_loss.item())
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            rho_s, _ = spearmanr(np.squeeze(pred_batch_numpy), np.squeeze(labels_batch_numpy))
            mean_srocc.append(rho_s)
            rho_p, _ = pearsonr(np.squeeze(pred_batch_numpy), np.squeeze(labels_batch_numpy))
            mean_plcc.append(rho_p)
    res['sum_abs'] = sum_losses
    res['sum_mape'] = mape
    res['srocc'] = mean_srocc
    res['plcc'] = mean_plcc
    df = pd.DataFrame(res)
    df.to_excel(f'data_out/{model_name}.xlsx', index=False)
    sum_loss = np.mean(sum_losses)
    print(check)
    print('test / loss:%f / mape:%f /SROCC:%4f / PLCC:%4f' % (sum_loss, np.mean(mape), np.mean(mean_srocc), np.mean(mean_plcc)))