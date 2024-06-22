import os
import torch
from torch.utils.data import DataLoader
from model.coatnet import get_my_coatnet

from config import MyConfig
from Data_Loader import chest_dataset
import numpy as np
from trainer import train_epoch, eval_epoch

from torch.utils.tensorboard import SummaryWriter
import warnings

# 禁用特定警告
warnings.filterwarnings("ignore", category=UserWarning)
config = MyConfig().config
torch.manual_seed(config.random_seed)
# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')
model_name = config.model_classes[config.now_class]

print(f"now model is {model_name}")

train_data = os.path.join("../datasets", model_name, config.train_data)
label = os.path.join("../datasets", model_name, config.label)
test = os.path.join("../datasets", model_name, config.test_path)
Training_Data = chest_dataset(train_data, label,)
Testing_Data = chest_dataset(train_data, label, is_train=False)

net = get_my_coatnet().to(config.device)

num_train = len(Training_Data)
train_idx = Training_Data.index
valid_idx = Testing_Data.index
print(f'seed:{config.random_seed}')
np.random.seed(config.random_seed)

writer1 = SummaryWriter()

for fold in range(config.K):
    best_model = 1e5

    print('number of train scenes: %d' % len(train_idx))
    print('number of valid scenes: %d' % len(valid_idx))

    train_loader = DataLoader(dataset=Training_Data, batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=True
                              )
    valid_loader = DataLoader(dataset=Testing_Data, batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=True
                              )


    if config.checkpoint is not None:

         checkpoint = torch.load(config.checkpoint)
         net.load_state_dict(checkpoint['net'])
         best_model = checkpoint['loss']
         start_epoch = checkpoint['epoch']
         print(f"now is load checkpoint {config.checkpoint}")
         print(f"start epoch:{start_epoch}")
         print(f"best model:{best_model}")
    else:
        def xavier_init(m):
             if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                 torch.nn.init.xavier_uniform_(m.weight)
        net.apply(xavier_init)
        start_epoch = 0

    params = list(net.parameters())

    # loss function & optimization
    criterion = torch.nn.BCEWithLogitsLoss()
    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)



    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    v_loss = 0
    v_rho_s = 0
    v_rho_p = 0
    # train & validation
    for epoch in range(start_epoch, config.n_epoch):
        t_loss, t_rho_s, t_rho_p, bce_t= train_epoch(config, epoch, net, criterion,l1,
                                                       optimizer,
                                                       scheduler,
                                                       train_loader)

        if (epoch + 1) % config.val_freq == 0:
            v_loss, v_rho_s, v_rho_p, bce_v = eval_epoch(config, epoch, net,criterion,l1,
                                                  valid_loader)
            if v_loss < best_model:
                weights_file_name = f"aba1_{model_name}_K{fold + 1}_best_model.pth"
                weights_file = os.path.join(config.snap_path, model_name, weights_file_name)
                torch.save({
                    'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': v_loss
                }, weights_file)
                print(f"best model saved!loss:{v_loss}")
                best_model = v_loss
        writer1.add_scalars(f'{model_name} Loss', {"Train Loss": t_loss, "Valid Loss": v_loss},
                            epoch + 1 + fold * (config.n_epoch - start_epoch))
        writer1.add_scalars(f'SROCC', {"Train Loss": t_rho_s, "Valid Loss": v_rho_s},
                            epoch + 1 + +fold * (config.n_epoch - start_epoch))
        writer1.add_scalars(f'PLCC', {"Train Loss": t_rho_p, "Valid Loss": v_rho_p},
                            epoch + 1 + +fold * (config.n_epoch - start_epoch))
        writer1.add_scalars(f'BCE LOSS', {"Train Loss": bce_t, "Valid Loss": bce_v},
                            epoch + 1 + +fold * (config.n_epoch - start_epoch))


writer1.close()
