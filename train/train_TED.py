import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crz.io2 import MYDataset
from Final_erro_detetion_network import RESGCN_SPE_TRANSFORM_ADROP
from crz.utils_crz import FocalLoss, AdvancedDropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caculate_TP(pre, lab):
    lab1 = torch.argmax(lab, dim=1)
    pre1 = torch.argmax(pre, dim=1)
    # print(pre1.shape,torch.sum(lab1==1),torch.sum(pre1==1))
    # recall_true = (torch.sum((1-pre1)*(1-lab1))/((1-lab1).sum())).item()
    TP = torch.sum((pre1) * (lab1))
    FP = torch.sum((pre1) * (1 - lab1))
    FN = torch.sum((1 - pre1) * (lab1))
    TN = torch.sum((1 - pre1) * (1 - lab1))
    # print(recall_true,recall_fault)
    return TP.item(), FP.item(), FN.item(),TN.item()

class ERRO_DETECT_TRAINER():
    def __init__(self,train_dataset_path,
                      test_dataset_path,
                      train_epoch,
                      batch_size,
                      learning_rate,
                      layerdeep,
                      alpha,
                      gamma=0 ):
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        # self.model = GCN_SPE_TRANSFORM(3,device=device).to(device)
        # self.model = GCN_SPEADROP_TRANSFORM(3,device=device).to(device)
        self.model = RESGCN_SPE_TRANSFORM_ADROP(3,device=device).to(device)
        ############################
        # set optimizer
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
                #print(m)
                if hasattr(m, "weight") and m.weight is not None:
                    res_params.append(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    res_params.append(m.bias)
        #print(res_params)
        self.optimizer = torch.optim.Adam([
                               {'params': res_params, 'lr': learning_rate},
                               {'params': dp_params, 'lr': 1e-4}], weight_decay=5e-4)
        ############################
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch+1))
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_foc = FocalLoss(gamma=gamma,alpha=alpha)

        TRAIN_dataset = MYDataset(train_dataset_path,layerdeep)
        self.train_dataset = DataLoader(TRAIN_dataset,batch_size=1,num_workers=2)

        Test_dataset = MYDataset(test_dataset_path,layerdeep)
        self.test_dataset = DataLoader(Test_dataset,batch_size=1,num_workers=2)

    def train_step(self):
        epoch_loss = 0
        self.model.train()
        TP,FP,FN,TN = 0,0,0,0
        iter = 0
        self.optimizer.zero_grad()
        for swc_name,data in tqdm(self.train_dataset):
            data_len = len(data)
            grap_pre = torch.zeros([data_len,2])
            grap_lab = torch.zeros([data_len,2])
            for i in range(data_len):
                adj,swc_feat,spe_feat,lab,pos = data[i]
                adj,swc_feat,spe_feat,lab,pos = adj.to(device),swc_feat.to(device),spe_feat.to(device),lab.to(device),pos.to(device)
                # if lab[0, 1] == 0 and random.uniform(0, 1) < 0.5:
                #     continue
                iter += 1
                adj = adj.squeeze()
                swc_feat = swc_feat.squeeze()
                spe_feat = spe_feat.squeeze()
                #print(swc_feat)
                #print(i,adj.shape,swc_feat*255,lab.shape,spe_feat.shape)
                pre_result = self.model(adj, swc_feat, spe_feat,pos)
                #loss = self.criterion_bce(pre_result, lab)
                loss = self.criterion_foc(pre_result, lab)
                if loss.isnan():
                    print("loss nan")
                    return

                loss.backward()
                if iter % self.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                grap_pre[i] = pre_result[0].detach()
                grap_lab[i] = lab[0].detach()

            TP0, FP0, FN0, TN0 = caculate_TP(pre=grap_pre, lab=grap_lab)
            TP += TP0
            FP += FP0
            FN += FN0
            TN += TN0

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        F1 = 2 * precision * recall / (precision + recall + 1e-5)
        print("TP: ",TP,"FP: ",FP,'FN: ',FN, 'TN:',TN)
        return precision, recall, F1, epoch_loss/iter


    #def add_train_dataset(self,adj,swc_feat):

    def test(self):
        self.model.eval()
        test_loss = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        iter = 0
        for swc_name, data in tqdm(self.test_dataset):
            data_len = len(data)
            grap_pre = torch.zeros([data_len, 2])
            grap_lab = torch.zeros([data_len, 2])
            for i in range(data_len):
                iter += 1
                adj, swc_feat, spe_feat, lab, pos = data[i]
                adj, swc_feat, spe_feat, lab, pos = adj.to(device), swc_feat.to(device), spe_feat.to(device), lab.to(device), pos.to(device)

                adj = adj.squeeze()
                swc_feat = swc_feat.squeeze()
                spe_feat = spe_feat.squeeze()
                # print(i,adj.shape,swc_feat*255,lab.shape,spe_feat.shape)
                pre_result = self.model(adj, swc_feat, spe_feat,pos)
                loss = self.criterion_foc(pre_result.detach(), lab.detach())
                test_loss += loss.item()

                grap_pre[i] = pre_result[0].detach()
                grap_lab[i] = lab[0].detach()

            # print(torch.sum(grap_lab[:,1]))
            #loss = self.criterion_foc(grap_pre.to(device), grap_lab.to(device))
            TP0, FP0, FN0,TN0 = caculate_TP(pre=grap_pre, lab=grap_lab)
            TP += TP0
            FP += FP0
            FN += FN0
            TN += TN0
        test_loss /= iter

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        F1 = 2 * precision * recall / (precision + recall + 1e-5)
        # print("precision: ", precision, "recall: ", recall, "F1: ", F1)
        print("TP: ", TP, "FP: ", FP, 'FN: ', FN, 'TN: ',TN)
        return precision,recall,F1,test_loss

    def train(self):
        epoch_loss_min = 10
        save_path = os.path.join(r'F:\neuron_repair\crz\checkpoint3',model_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        #mlist = []
        for epoch in range(self.train_epoch):
            print('epoch: ', epoch)
            ################################### train #############################################################
            train_precision,train_recall,train_F1,epoch_loss = self.train_step()
            print('train precision1: ',train_precision,"recall: ",train_recall,'F1: ',train_F1,'loss: ',epoch_loss)
            if epoch_loss == torch.nan:
                break
            if epoch_loss_min>epoch_loss:
                torch.save(self.model.state_dict(),os.path.join(save_path,'TED_best.pkl'))
                epoch_loss_min = epoch_loss
            if (epoch+1)%10 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path,'TED_'+str(epoch)+'.pkl'))

            ################################## test ################################################################
            test_precision, test_recall,test_F1,test_loss = self.test()
            print('test precision: ', test_precision, "recall: ", test_recall,'F1: ',test_F1,'loss: ',test_loss)

            self.scheduler.step()
            save_txt(save_path,[train_precision,train_recall,train_F1,test_precision, test_recall,test_F1,epoch_loss,test_loss])
        # print("train_acc_max: ",self.train_acc)
        # print("test_acc_max: ",test_max_pre,test_max_rec)

def save_txt(path,data):
    path = path+'/'+model_name+'.txt'
    with open(path, "a") as file:
        # 遍历列表中的元素并将它们写入文件
        for item in data:
            file.write(str(item))
            file.write(" ")
        file.write('\n')

def test_only():

    model = RESGCN_SPE_att_adrop(3)
    #model = RESGCN_SPE_att(3)
    # torch.manual_seed(1)
    checkpoint = torch.load(r'F:\neuron_repair\crz\checkpoint2\RESGCN5_SPE16_att3_adrop2_1\TED_best.pkl',weights_only=False)
    #model = checkpoint#.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # for name, parms in model.named_parameters():
    #     #if 'BatchNorm' in name:
    #         print('-->name:', name)
    #         print('-->value:', parms)
    #         #print('-->grad_value:', parms.grad[0])

    test_dataset_path = r"F:\neuron_repair\dataset\sp_16x16_2\test_dataset"
    Test_dataset = MYDataset(test_dataset_path)
    test_dataset = DataLoader(Test_dataset, batch_size=1, num_workers=1)
    TP = 0
    FP = 0
    FN = 0

    for swc_name, data in tqdm(test_dataset):
        data_len = len(data)
        grap_pre = torch.zeros([data_len, 2])
        grap_lab = torch.zeros([data_len, 2])
        #print(swc_name)

        for i in range(data_len):
            adj, swc_feat, spe_feat, lab = data[i]
            adj, swc_feat, spe_feat, lab = adj.to(device), swc_feat.to(device), spe_feat.to(device), lab.to(device)

            adj = adj.squeeze()
            swc_feat = swc_feat.squeeze()
            # lab = lab.squeeze()
            spe_feat = spe_feat.squeeze()
            # print(i,adj.shape,swc_feat*255,lab.shape,spe_feat.shape)
            pre_result = model(adj, swc_feat, spe_feat)

            grap_pre[i] = pre_result[0].detach()
            grap_lab[i] = lab[0].detach()

        #print(torch.sum(grap_lab[:,1]))
        #loss = self.criterion_foc(grap_pre.to(device),grap_lab.to(device))
        # precision0,recall0,F10 = self.caculate_acc(pre=grap_pre, lab=grap_lab)
        # precision += precision0 / len(self.test_dataset)
        # recall += recall0 / len(self.test_dataset)
        # F1 += F10 / len(self.test_dataset)
        TP0, FP0, FN0 = caculate_TP(pre=grap_pre,lab=grap_lab)
        TP += TP0
        FP += FP0
        FN += FN0
        #test_loss += loss.item()/len(self.test_dataset)

    precision = TP/(TP+FP + 1e-5)
    recall = TP/(TP+FN + 1e-5)
    F1 = 2*precision*recall/(precision+recall+1e-5)
    print("precision: ",precision,"recall: ",recall,"F1: ",F1)

if __name__ == '__main__':
    model_name = 'RESGCN_SPE4_TRANSFORM_ADROP3_L4_p22'#'test'

    # train_dataset_path =  r'F:\neuron_repair\dataset\train_data_gcn'
    # test_dataset_path = r'F:\neuron_repair\dataset\test_data_gcn'
    train_dataset_path = r"F:\neuron_repair\dataset\sp_16x16_2\train_dataset"
    test_dataset_path = r"F:\neuron_repair\dataset\sp_16x16_2\test_dataset"
    trainer = ERRO_DETECT_TRAINER(train_dataset_path = train_dataset_path,
                                  test_dataset_path = test_dataset_path,
                                  train_epoch = 60,
                                  batch_size = 64,
                                  learning_rate=0.001,
                                  layerdeep=4,
                                  alpha = 5,
                                  gamma= 0
                                  )
    trainer.train()
    #test_only()