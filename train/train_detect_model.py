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
from dataset.utils import loadswc, saveswc
from model.PointNet import PointNet
from model.DGCNN import DGCNN
from model.GCN import GCN
from dataset.detect_dataset import DetectDataset
from model.check_model import TED, AdvancedDropout
from model.pointnet_utils import feature_transform_reguliarzer

current_time = datetime.now()
print('Current time: {}'.format(current_time))
bar_format = f"{Fore.WHITE}{'{l_bar}'}{Fore.GREEN}{'{bar}'}{Fore.WHITE}|{'{n_fmt}'}/{'{total_fmt}'}\t[{'{elapsed}'}<{'{remaining}'},{'{rate_fmt}'}]{'{postfix}'}"


def parse_args():
    parser = argparse.ArgumentParser(description='Train for Detect')
    parser.add_argument('--is_test', action="store_true", default=False, help='test or train')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--lr', type=float, default=0.001, help='generator learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--loss_type', type=str, default='Fl', choices=['Fl', 'Bec', 'Nl'])
    parser.add_argument('--alpha', type=float, default=5, help='Fl alpha')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--train_root', default=r'../data/error_detect_data/train', type=str,
                        help='path to train dataset')
    parser.add_argument('--test_root', default=r'../data/error_detect_data/test', type=str, help='path to test dataset')
    parser.add_argument('--save_dir', type=str, default=r'../run/Error_Detect', help='path to save')
    parser.add_argument('--model_name', type=str, default='TED',
                        choices=['GCN', 'DGCNN', 'Pointnet', 'TED'])
    args = parser.parse_args()
    return args


args = parse_args()

model_dict = {
    'GCN': GCN(),
    'Pointnet': PointNet(),  # PointNet 使用Nl_loss
    'DGCNN': DGCNN(),
    'TED': TED(device=args.device)
}


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.model = model_dict[self.args.model_name].to(self.args.device)
        print('model: ', args.model_name)
        if self.args.model_name == 'TED':
            dp_params = []
            res_params = []
            for m in self.model.modules():
                if isinstance(m, AdvancedDropout):
                    dp_params.append(m.weight_h)
                    dp_params.append(m.bias_h)
                    dp_params.append(m.weight_mu)
                    dp_params.append(m.bias_mu)
                    dp_params.append(m.weight_sigma)
                    dp_params.append(m.bias_sigma)
                else:
                    # print(m)
                    if hasattr(m, "weight") and m.weight is not None:
                        res_params.append(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        res_params.append(m.bias)
                # print(res_params)
            self.optimizer = torch.optim.Adam([
                {'params': res_params, 'lr': args.lr},
                {'params': dp_params, 'lr': 1e-4}], weight_decay=self.args.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if not self.args.is_test:
            self.train_dataset = DetectDataset(self.args.train_root)
            self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.test_dataset = DetectDataset(self.args.test_root)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        if not self.args.is_test:
            self.save_root = os.path.join(self.args.save_dir, str(self.args.model_name),
                                          current_time.strftime("%Y-%m-%d-%H-%M"))
            self.results_root = os.path.join(self.save_root, 'result', current_time.strftime("%Y-%m-%d-%H-%M"))
            os.makedirs(self.results_root, exist_ok=True)
            self.log_file = os.path.join(self.save_root, 'log.txt')
            self._log_save(is_start=True)

    def run_step(self, loader, epoch=None, mode='train', best_recall=None, best_precision=None, best_f1=None):
        loss_list = []
        TP, FP, TN, FN = 0, 0, 0, 0
        if mode == 'train':
            desc = f'Train: {epoch + 1:04d}'
        elif mode == 'valid':
            desc = f'Valid: {epoch + 1:04d}'
        elif mode == 'test':
            desc = f'Test: '
        loop_loader = tqdm(enumerate(loader), total=len(loader), desc=desc, bar_format=bar_format)
        iter = 0
        for _, data in loop_loader:
            swc_name, data = data
            for i in range(len(data)):
                # print(len(data))
                adj, feat_xyz, feat_sp, label = data[i]
                feat_xyz = feat_xyz.to(self.args.device)
                label = label.to(self.args.device)
                adj = adj.to(self.args.device)
                feat_sp = feat_sp.squeeze(dim=0).to(self.args.device)
                output = self.model(feat_xyz, adj, feat_sp)
                loss = self._calculate_loss(output, label, self.args.loss_type)
                if loss.isnan():
                    print('nan loss')
                    continue
                loss_list.append(loss.item())
                if mode == 'train':
                    iter += 1
                    loss.backward()
                    if iter % 64 == 0 and self.args.model_name == 'TED':
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    elif self.args.model_name != 'TED':
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                metric = self._calculate_metrics(output, label)
                TP += metric['TP'].cpu()
                TN += metric['TN'].cpu()
                FP += metric['FP'].cpu()
                FN += metric['FN'].cpu()
            Precision = TP / (TP + FP + 1e-5)
            Recall = TP / (TP + FN + 1e-5)
            f1_score = 2 * Precision * Recall / (Precision + Recall + 1e-5)
            info = f"loss: {np.mean(loss_list):.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, 'F1:' {f1_score:.4f}"
            if best_f1 is not None:
                if f1_score > best_f1:
                    loop_loader.set_postfix_str(f"{Fore.LIGHTGREEN_EX}{info}")
                else:
                    loop_loader.set_postfix_str(f"{Fore.LIGHTWHITE_EX}{info}")
            else:
                loop_loader.set_postfix_str(f"{Fore.LIGHTWHITE_EX}{info}")
        return info

    def train(self):
        best_precision = 0
        best_recall = 0
        best_epoch = 0
        best_f1 = 0
        best_loss = 1e10
        best_model = self.model
        for epoch in range(self.args.epochs):
            self.model.train()
            train_info = self.run_step(self.train_loader, epoch=epoch)
            self.scheduler.step()
            self.model.eval()
            with torch.no_grad():
                valid_info = self.run_step(self.test_loader, epoch=epoch, mode='valid', best_recall=best_recall,
                                           best_precision=best_precision, best_f1=best_f1)
            train_metric = self.extract_metrics(train_info)
            valid_metric = self.extract_metrics(valid_info)
            self._log_save(log=f"Train_{epoch + 1}: " + train_info)
            self._log_save(log=f"Valid_{epoch + 1}: " + valid_info + "\n")
            if train_metric['loss'] < best_loss:
                best_epoch = epoch
                best_loss = train_metric['loss']
                best_precision = valid_metric['precision']
                best_recall = valid_metric['recall']
                best_f1 = valid_metric['f1']
                best_model = self.model.state_dict()
            if (epoch + 1) % 2 == 0:
                save_path = os.path.join(self.save_root, 'checkpoint', 'epoch_' + str(epoch + 1) + '.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
        best_model_save_path = os.path.join(self.save_root, 'checkpoint', 'best_epoch_' + str(best_epoch + 1) + '.pth')
        torch.save(best_model, best_model_save_path)
        best_log = f"Best Model: Epoch {best_epoch + 1}, Best Precision: {best_precision:.4f}, Best_Recall: {best_recall}, Best F1: {best_f1}\n"
        self._log_save(log=best_log)

    def test(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        # self.save_root = checkpoint_path.replace('checkpoint', 'result', current_time.strftime("%Y-%m-%d-%H-%M"))
        # self.log_file = os.path.join(self.save_root, 'test_log.txt')
        self.model.eval()
        with torch.no_grad():
            test_info = self.run_step(self.test_loader, mode='test')
            # self._log_save(log=f"Test: " + test_info)

    def _calculate_loss(self, output, target: torch.Tensor, loss_fn='Fl') -> torch.Tensor:
        def FocalLoss(pred, label, gamma=0, alpha=10) -> torch.Tensor:
            BCE_loss = F.binary_cross_entropy_with_logits(pred, label)
            pt = torch.exp(-BCE_loss)

            alpha_t = 1
            if label[:, 1] == 1:
                alpha_t = alpha
            FL_loss = alpha_t * (1 - pt) ** gamma * BCE_loss
            if FL_loss.mean().isnan():
                print(BCE_loss)

            return FL_loss.mean()

        def NllLoss(output, target) -> torch.Tensor:
            # print('output shape: ', output[0].shape)
            # print('target shape: ', target.shape)
            # print('output: ', output[0])
            target = target.argmax(dim=1)
            loss = F.nll_loss(output[0], target)
            mat_diff_loss = feature_transform_reguliarzer(output[1])

            total_loss = loss + mat_diff_loss * 0.001
            return total_loss

        if loss_fn == 'Fl':
            if type(output) == tuple:
                output = output[0]
            return FocalLoss(output, target, gamma=0, alpha=self.args.alpha)
        elif loss_fn == 'Bec':
            if type(output) == tuple:
                output = output[0]
            return nn.BCEWithLogitsLoss()(output, target)
        elif loss_fn == 'Nl':
            return NllLoss(output, target)

    def _calculate_metrics(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        if type(output) == tuple:
            output = output[0]
        else:
            output = torch.sigmoid(output).round()
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)

        TP, FP, TN, FN = 0, 0, 0, 0
        TP += torch.sum(torch.logical_and(output.eq(1), target.eq(1)))
        FP += torch.sum(torch.logical_and(output.eq(1), target.eq(0)))
        TN += torch.sum(torch.logical_and(output.eq(0), target.eq(0)))
        FN += torch.sum(torch.logical_and(output.eq(0), target.eq(1)))
        return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

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
                # f.write(
                #     f"lr_scheduler: {self.scheduler.__class__.__name__}, step_size={self.scheduler.step_size}, gamma={self.scheduler.gamma}\n")
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
    # args.train_root = r'../data/data_for_detect/train'
    # args.test_root = r'../data/data_for_detect/test'
    # args.save_dir = r'../run/Error_Detect'
    trainer = Trainer(args)
    trainer.train()

    # trainer.test(r'F:\Neruron_Repair\Neuron_Detect_Repair_upload\test\check_point\TED_best.pkl')
