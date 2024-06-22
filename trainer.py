import os

import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr




def train_epoch(config, epoch, net, criterion, l1, optimizer, scheduler, train_loader,
                p=False):
    losses = []
    bcelosses = []
    net.train()
    mean_srocc = []
    mean_plcc = []
    for data in tqdm(train_loader):
        d_img_org_0 = data['d_img_org'][0].to(config.device)
        d_img_org_1 = data['d_img_org'][1].to(config.device)
        optimizer.zero_grad()
        pred_tag1, pred, mirror0 = net(d_img_org_0, d_img_org_1)
        labels_tag = data['score_tag'].to(config.device)
        labels = data['score'].to(config.device)
        bceloss = criterion(pred_tag1, labels_tag.float())
        bcelosses.append(bceloss.item())
        l1loss = l1(pred, labels)
        losses.append(l1loss.item())
        loss = bceloss
        loss.backward()
        optimizer.step()
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        for i in range(pred_batch_numpy.shape[0]):
            rho_s, _ = spearmanr(np.squeeze(pred_batch_numpy[i]), np.squeeze(labels_batch_numpy[i]))
            mean_srocc.append(rho_s)
            rho_p, _ = pearsonr(np.squeeze(pred_batch_numpy[i]), np.squeeze(labels_batch_numpy[i]))
            mean_plcc.append(rho_p)
    scheduler.step()

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f / lr:%f' % (
        epoch + 1, np.mean(losses).item(), np.mean(mean_srocc), np.mean(mean_plcc), scheduler.get_last_lr()[0]))
    # save weights
    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = "sam_epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, config.model_classes[config.now_class], weights_file_name)
        torch.save({
            'epoch': epoch,
            'net': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses), rho_s, rho_p, np.mean(bcelosses)


""" validation """


def eval_epoch(config, epoch, sam, net, criterion, l1, test_loader):
    with torch.no_grad():
        losses = []
        bcelosses = []
        mean_srocc = []
        mean_plcc = []
        sum_ = []
        net.eval()
        for data in tqdm(test_loader):
            d_img_org_0 = data['d_img_org'][0].to(config.device)
            d_img_org_1 = data['d_img_org'][1].to(config.device)

            pred_tag, pred, mirror= net(d_img_org_0, d_img_org_1)
            labels_tag = data['score_tag'].to(config.device)
            labels = data['score'].to(config.device)
            bceloss = criterion(pred_tag, labels_tag.float())
            bcelosses.append(bceloss.item())
            l1loss = l1(pred, labels)
            losses.append(l1loss.item())
            sum_.append(l1(pred.sum(dim=1), labels.sum(dim=1)).item())
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            for i in range(pred_batch_numpy.shape[0]):
                rho_s, _ = spearmanr(np.squeeze(pred_batch_numpy[i]), np.squeeze(labels_batch_numpy[i]))
                mean_srocc.append(rho_s)
                rho_p, _ = pearsonr(np.squeeze(pred_batch_numpy[i]), np.squeeze(labels_batch_numpy[i]))
                mean_plcc.append(rho_p)
        loss = np.mean(losses)
        print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (
        epoch + 1, loss, np.mean(mean_srocc), np.mean(mean_plcc)))
        print('sum loss:', np.mean(sum_))

    return np.mean(sum_), rho_s, rho_p, np.mean(bcelosses)
