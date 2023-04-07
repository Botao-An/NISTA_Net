import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
import joblib
import progressbar
import argparse
from torch.utils.data import TensorDataset, DataLoader
# import torch.utils.data as Data

# 定义训练函数
def train_model(model, args, data_dim=1):
    # 数据准备
    data_dict = joblib.load(args.data_path)
    if data_dim==1:
        train = TensorDataset(torch.Tensor(data_dict['train_1d']), torch.Tensor(data_dict['train_label']))
        test  = TensorDataset(torch.Tensor(data_dict['test_1d']),  torch.Tensor(data_dict['test_label']))
    elif data_dim==2:
        train = TensorDataset(torch.Tensor(data_dict['train_2d']), torch.Tensor(data_dict['train_label']))
        test  = TensorDataset(torch.Tensor(data_dict['test_2d']),  torch.Tensor(data_dict['test_label']))
    else:
        raise ValueError("Unexptected data dimension number, opt: 1, 2")

    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_works)
    test_loader  = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=True, num_workers=args.n_works)

    # 判断CUDA是否可用
    if args.use_cuda:
        model = model.cuda()

    # 设定并创建工作目录
    project_path = './results/'+args.project_name+'/'
    outf = project_path+'model/'

    # 选定优化器
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise Exception("optimizer not implement")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step,gamma=args.gamma)

    EPOCH = args.n_epochs
    Loss_train = np.zeros((EPOCH,))
    Loss_test = np.zeros((EPOCH,))
    Acc_test = np.zeros((EPOCH,))
    Acc_train = np.zeros((EPOCH,))

    bar = progressbar.ProgressBar(maxval=EPOCH-1).start()  #进度条显示函数

    if not os.path.exists(outf):
        os.makedirs(outf)
    with open(project_path+"train_log.txt", "w") as f:
        # 写入表头信息
        f.write("Project Name: %s"  % (args.project_name))
        f.write('\n')
        f.write("Time: %s"  % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        f.write('\n')
        f.write("epoch: %03d, batch_size: %03d, optimizer: %s, learning_rate: %.6f, momentum: %.3f, weight_decay: %.6f, steps: %03d, gamma: %.3f" 
            % (EPOCH, args.batch_size, args.opt, args.lr, args.momentum, args.weight_decay, args.step, args.gamma))
        f.write('\n')
        f.write('\n')
        f.flush()
        # 开始多个epoch
        for epoch in range(EPOCH): 
            bar.update(epoch)
            model.train()
            
            correct = 0
            train_loss = 0
            rec_error = 0
            for step, (x, y) in enumerate(train_loader): # batct_size循环
                b_x = Variable(x)
                b_y = Variable(y)
                if args.use_cuda:
                    b_y, b_x = b_y.cuda(), b_x.cuda()
    #             print('input size: ',b_x.size())
                b_x = b_x.float()
                scores = model(b_x) #scores = model(b_x)
                loss = F.nll_loss(scores, b_y.long())      # negative log likelyhood
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                optimizer.step()                    # apply gradients
                model.zero_grad()
                # computing training stats
                pred = scores.data.max(1, keepdim=True)[1]
                correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()
                train_loss += F.nll_loss(scores, b_y.long(), reduction='sum').data.item()

            Acc_train[epoch] =  100 * float(correct) / float(len(train_loader.dataset))
            Loss_train[epoch] = train_loss / len(train_loader.dataset)

            # testing
            model.eval()
            correct = 0
            test_loss = 0
            NNZ = 0.0
            for step, (x, y) in enumerate(test_loader):
                b_x = Variable(x)   
                b_y = Variable(y)               # batch label
                if args.use_cuda:
                    b_y, b_x = b_y.cuda(), b_x.cuda()
                b_x = b_x.float()
                scores = model(b_x)
                b_y = b_y.long()
                test_loss += F.nll_loss(scores, b_y.long(), reduction='sum').data.item()
                pred = scores.data.max(1, keepdim=True)[1]
                correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()

            Loss_test[epoch] = test_loss/len(test_loader.dataset)
            Acc_test[epoch] =  100 *  float(correct) /float(len(test_loader.dataset))
            print('Epoch: ', epoch, '| train loss: ', Loss_train[epoch],'| train acc: ', Acc_train[epoch], '%','| test acc: ', Acc_test[epoch], '%')
            print('Saving model......')
            torch.save(model.state_dict(), '%s/model_epo_%03d_acc_%.4f.pth' % (outf, epoch + 1, Acc_test[epoch]))
            f.write("EPOCH = %03d, Train_Loss: %.8f%%, Train_Acc: %.4f%%, Test_Acc: %.4f%%" % (epoch + 1, Loss_train[epoch], Acc_train[epoch], Acc_test[epoch]))
            f.write('\n')
            f.flush()
            scheduler.step()
    f.close()
    return model,Acc_train,Acc_test