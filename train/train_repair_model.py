import argparse
import os
import re
from datetime import datetime
import numpy as np
import torch
from colorama import Fore
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from dataset.repair_dataset import RepairDataset
from dataset.utils import loadswc, saveswc, load_SPfeature
from model.check_model import TLP
from model.GAE import GAE, VGAE


current_time = datetime.now()
print('Current time: {}'.format(current_time))
bar_format = f"{Fore.WHITE}{'{l_bar}'}{Fore.GREEN}{'{bar}'}{Fore.WHITE}|{'{n_fmt}'}/{'{total_fmt}'}\t[{'{elapsed}'}<{'{remaining}'},{'{rate_fmt}'}]{'{postfix}'}"


def parse_args():
    parser = argparse.ArgumentParser(description='Trian for Repair')
    parser.add_argument('--is_test', action="store_true", default=False, help='test or train')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--lr', type=float, default=0.001, help='generator learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--loss_type', type=str, default='Bec_w', choices=['Bec_w', 'Bec', 'Bec_d'])
    parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
    parser.add_argument('--train_root', type=str, default=r'../data/link_pred_data/train',
                        help='path to train dataset')
    parser.add_argument('--test_root', type=str, default=r'../data/link_pred_data/test', help='path to test dataset')
    parser.add_argument('--save_dir', type=str, default=r'../run/Repair', help='path to save')
    parser.add_argument('--model_name', type=str, default='TLP', choices=['TLP', 'gae', 'vgae'], help='model name')
    args = parser.parse_args()
    return args


args = parse_args()

model_dict = {
    'TLP': TLP(),
    'gae': GAE(),
    'vgae': VGAE()
}


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.model = model_dict[self.args.model_name].to(self.args.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20)

        if not self.args.is_test:
            self.train_dataset = RepairDataset(self.args.train_root, is_aug=False)
            self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
            print(f'Train dataset size: {len(self.train_dataset)}')
        self.test_dataset = RepairDataset(self.args.test_root, is_aug=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        print(f'Test dataset size: {len(self.test_dataset)}')
        if not self.args.is_test:
            self.save_root = os.path.join(self.args.save_dir,
                                          str(self.args.model_name),
                                          current_time.strftime("%Y-%m-%d-%H-%M"))
            self.results_root = os.path.join(self.save_root, 'result', current_time.strftime("%Y-%m-%d-%H-%M"))
            os.makedirs(self.results_root, exist_ok=True)
            self.log_file = os.path.join(self.save_root, 'log.txt')
            self._log_save(is_start=True)

    def run_step(self, model, loader, epoch=None, mode='train', best_recall=None, best_precision=None, best_f1=None,
                 save_root=None):
        loss_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        # TP, FP, TN, FN = 0, 0, 0, 0
        if mode == 'train':
            desc = f'Train: {epoch + 1:04d}'
        elif mode == 'valid':
            desc = f'Valid: {epoch + 1:04d}'
        elif mode == 'test':
            desc = f'Test: '
        loop_loader = tqdm(enumerate(loader), total=len(loader), desc=desc, bar_format=bar_format)
        for _, data in loop_loader:
            # feat_sp, feat_xyz, adj, label = data
            feat_xyz, label, adj, path = data
            feat_sp = load_SPfeature(path[0])
            feat_xyz = feat_xyz.to(self.args.device)
            label = label.to(self.args.device)
            adj = adj.to(self.args.device)
            feat_sp = feat_sp.squeeze(dim=0)
            feat_xyz = feat_xyz.squeeze(dim=0)
            adj = adj.squeeze(dim=0)
            label = label.squeeze(dim=0)

            feat_sp = feat_sp.to(self.args.device)

            output = model(feat_xyz, adj, feat_sp)
            loss = self._calculate_loss(output, label, self.args.loss_type)
            loss_list.append(loss.item())

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # if mode == 'test':
            #     self._deal_output(output, swc_path=path, save_root=save_root)

            metric = self._calculate_metrics(output, label)
            precision_list.append(metric['precision'].cpu().numpy())
            recall_list.append(metric['recall'].cpu().numpy())
            f1_score_list.append(metric['f1'].cpu().numpy())
            info = f"loss: {np.mean(loss_list):.4f}, Precision: {np.mean(precision_list):.4f}, Recall: {np.mean(recall_list):.4f}, F1: {np.mean(f1_score_list):.4f}"
            if best_f1 is not None:
                if np.mean(f1_score_list) > best_f1:
                    # print('f1 666')
                    loop_loader.set_postfix_str(f"{Fore.LIGHTGREEN_EX}{info}")
                else:
                    loop_loader.set_postfix_str(f"{Fore.LIGHTWHITE_EX}{info}")
            else:
                loop_loader.set_postfix_str(f"{Fore.LIGHTWHITE_EX}{info}")
        return model, info

    def train(self):
        best_precision = 0
        best_recall = 0
        best_f1 = 0
        best_epoch = 0
        best_loss = 1e10
        best_model = self.model
        for epoch in range(self.args.epochs):
            self.model.train()
            self.model, train_info = self.run_step(self.model, self.train_loader, epoch=epoch)
            self.scheduler.step()
            self.model.eval()
            with torch.no_grad():
                self.model, valid_info = self.run_step(self.model, self.test_loader, epoch=epoch, mode='valid',
                                                       best_recall=best_recall,
                                                       best_precision=best_precision, best_f1=best_f1)
            train_metric = self.extract_metrics(train_info)
            valid_metric = self.extract_metrics(valid_info)
            self._log_save(log=f"Train_{epoch + 1}: " + train_info)
            self._log_save(log=f"Valid_{epoch + 1}: " + valid_info + "\n")
            if train_metric['loss'] < best_loss:
                best_precision = valid_metric['precision']
                best_recall = valid_metric['recall']
                best_f1 = valid_metric['f1']
                best_epoch = epoch
                best_model = self.model.state_dict()
                best_model_save_path = os.path.join(self.save_root, 'checkpoint', 'best_epoch' + '.pth')
                os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                torch.save(best_model, best_model_save_path)
            if (epoch + 1) % 2 == 0:
                save_path = os.path.join(self.save_root, 'checkpoint', 'epoch_' + str(epoch + 1) + '.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                # self.test(save_path)
        best_log = f"Best Model: Epoch {best_epoch + 1}, Best Precision: {best_precision:.4f}, Best_Recall: {best_recall}"
        self._log_save(log=best_log)

    def test(self, checkpoint_path):
        test_model = model_dict[self.args.model_name].to(self.args.device)
        test_model.load_state_dict(torch.load(checkpoint_path))
        # self.save_root = checkpoint_path.replace('checkpoint', 'result')
        # self.save_root = os.path.join(self.save_root, current_time.strftime("%Y-%m-%d-%H-%M"))
        # self.log_file = os.path.join(self.save_root, 'test_log.txt')
        test_model.eval()
        with torch.no_grad():
            test_model, test_info = self.run_step(test_model, self.test_loader, mode='test')
            # self._log_save(log=f"Test: " + test_info)

    def _calculate_loss(self, output, target: torch.Tensor, loss_fn='Bec_w') -> torch.Tensor:
        def dice_loss(pred, label):
            pred = pred.squeeze(dim=0)
            label = label.squeeze(dim=0)
            num = pred.shape[0]
            pred = pred.view(num, -1)
            label = label.view(num, -1)
            intersection = (pred * label).sum(-1).sum()
            union = (pred + label).sum(-1).sum()
            score = 1 - 2 * (intersection + 1e-5) / (union + 1e-5)
            return score

        def bce_with_dynamic_weight(output: torch.Tensor, target):
            output = torch.sigmoid(output)
            pos_weight = float(target.shape[1] * target.shape[1] - target.sum()) / target.sum()
            weight_mask = torch.where(target == 1)
            weight_tensor = torch.ones_like(target)
            weight_tensor[weight_mask] = pos_weight
            loss = F.binary_cross_entropy(output, target, weight=weight_tensor)
            return loss

        def bec_with_kl(output, target):
            n_nodes = target.shape[1]
            z_mean = output[1]
            z_log_var = output[2]
            # print('output: ', output[0])
            output = torch.sigmoid(output[0])
            # print('output: ', output)
            pos_weight = float(target.shape[1] * target.shape[1] - target.sum()) / target.sum()
            weight_mask = torch.where(target == 1)
            weight_tensor = torch.ones_like(target)
            weight_tensor[weight_mask] = pos_weight
            loss = F.binary_cross_entropy(output, target, weight=weight_tensor)
            KLD = -0.5 / n_nodes * torch.mean(torch.sum(
                1 + 2 * z_log_var - z_mean.pow(2) - z_log_var.exp().pow(2), 1))
            # print('loss: ', loss)
            # print('KLD: ', KLD)
            return loss + KLD

        def bce_with_dice(output: torch.Tensor, target: torch):
            output = torch.sigmoid(output)
            pos_weight = float(target.shape[1] * target.shape[1] - target.sum()) / target.sum()
            weight_mask = torch.where(target == 1)
            weight_tensor = torch.ones_like(target)
            weight_tensor[weight_mask] = pos_weight
            dice = dice_loss(output, target)
            loss = F.binary_cross_entropy(output, target, weight=weight_tensor)
            return loss + dice

        if self.args.model_name == 'vgae':
            return bec_with_kl(output, target)
        else:
            if loss_fn == 'Bec_w':
                return bce_with_dynamic_weight(output, target)
            elif loss_fn == 'Bec':
                return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, 10]).to(output.device))(output, target)
            elif loss_fn == 'Bec_d':
                return bce_with_dice(output, target)

    def _calculate_metrics(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        TP, FP, TN, FN = 0, 0, 0, 0
        if self.args.model_name == 'vgae':
            output = torch.sigmoid(output[0])
        else:
            output = torch.sigmoid(output)
        output = output.squeeze(dim=0)
        target = target.squeeze(dim=0)
        idx = torch.where(output >= 0.99)
        result = torch.zeros_like(output)
        result[idx[0], idx[1]] = 1
        TP += torch.sum(torch.logical_and(result.eq(1), target.eq(1)))
        FP += torch.sum(torch.logical_and(result.eq(1), target.eq(0)))
        TN += torch.sum(torch.logical_and(result.eq(0), target.eq(0)))
        FN += torch.sum(torch.logical_and(result.eq(0), target.eq(1)))

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _deal_output(self, output: torch.Tensor, swc_path: str, save_root: str, adj: torch.Tensor = None) -> None:
        pred = output.squeeze(dim=0)
        pred = pred.cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        det_idx = np.where(pred == 1)
        swc = loadswc(swc_path[0])
        if len(det_idx) > 0:
            print(swc_path[0])
            print(det_idx)
            swc = np.delete(swc, det_idx, axis=0)
        filename = os.path.basename(swc_path[0])
        save_path = os.path.join(save_root, filename)
        saveswc(save_path, swc)

    def _log_save(self, is_start=False, log=None) -> None:
        if is_start:
            with open(self.log_file, 'w') as f:
                f.write(f"Learning Rate: {self.args.lr}\n")
                f.write(f"Epochs: {self.args.epochs}\n")
                f.write(f"Batch Size: {self.train_loader.batch_size}\n")
                f.write(f"Weight Decay: {self.args.weight_decay}\n")
                f.write(f"Model Architecture: {self.model.__class__.__name__}\n")
                f.write(f"Loss Function: {self.args.loss_type}\n")
                f.write(f"Optimizer: {self.optimizer.__class__.__name__}\n")
                f.write(
                    f"lr_scheduler: {self.scheduler.__class__.__name__}, step_size={self.scheduler.step_size}, gamma={self.scheduler.gamma}\n")
                f.write(f"Data_root: {self.args.train_root}, {self.args.test_root}\n")
                f.write("Start training...\n")
        else:
            with open(self.log_file, 'a') as f:
                f.write(log)
                f.write("\n")

    def extract_metrics(self, info):
        numbers = re.findall(r"\d+\.\d+", info)  # 兼容任意小数位
        return {
            "loss": float(numbers[0]),
            "precision": float(numbers[1]),
            "recall": float(numbers[2]),
            "f1": float(numbers[3])
        }


if __name__ == "__main__":
    # args.model_name = 'Pointnet'
    # args.loss_type = 'Nl'
    # args.train_root = r'../data/data_for_repair/train'
    # args.test_root = r'../data/data_for_repair/test'
    # args.save_dir = r'../run/Error_Detect'
    args.is_test = False
    trainer = Trainer(args)
    if args.is_test:
        ckpt_path = r'../test/check_point/TLP.pth'
        trainer.test(ckpt_path)
    else:
        trainer.train()
