# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import argparse
import numpy as np
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from train import run_simple, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2
from model import TrajPreSimple,TrajPreAttnAvgLongUser,TrajPreLocalAttnLong


def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  cluster_emb_size=args.cluster_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path,
                                  use_cuda=args.use_cuda
                                  )
    argv = {
        'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 
        'voc_emb_size': args.voc_emb_size,'tim_emb_size': args.tim_emb_size, 
        'cluster_emb_size':args.cluster_emb_size,'hidden_size': args.hidden_size,
        'dropout_p': args.dropout_p, 'data_name': args.data_name,
        'learning_rate': args.learning_rate,'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu','optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
    print('*' * 15 + 'start training' + '*' * 15)
    print('model_mode:{} history_mode:{} users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    if parameters.model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters=parameters) #eliminate .cuda() 构建模型
        if parameters.use_cuda:
            model=model.cuda()
    elif parameters.model_mode == 'attn_avg_long_user': #this is final model
        model = TrajPreAttnAvgLongUser(parameters=parameters)    #delete cuda
        if parameters.use_cuda:
            model=model.cuda()
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters)  #delete cuda
        if parameters.use_cuda:
            model=model.cuda()
    if args.pretrain == 1:
        model.load_state_dict(torch.load("../pretrain/" + args.model_mode + "/res.m"))

    if 'max' in parameters.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in parameters.model_mode:
        parameters.history_mode = 'avg' 
    else:
        parameters.history_mode = 'whole'   #default

    if parameters.use_cuda:
        criterion = nn.NLLLoss().cuda() #delete cuda   NLLLoss=The negative log likelihood loss
    else:
        criterion = nn.NLLLoss() #delete cuda   NLLLoss=The negative log likelihood loss

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
    #                        weight_decay=parameters.L2)  #filter 得到requires_grad=True的参数,即可调参数
    optimizer = optim.Adam(model.parameters(), lr=parameters.lr,
                           weight_decay=parameters.L2)  #filter 得到requires_grad=True的参数,即可调参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-3)    #learning rate schedule,当学习停滞时减小lr，new_lr = lr * factor
        #max到底是说检测到的变化的参数数量不再增加还是说loss指标不再增加 This scheduler reads a metrics quantity
    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}

    candidate = parameters.data_neural.keys()   #candidate指的是什么loc?user? it is user candidate
    
    avg_acc_markov, users_acc_markov = markov(parameters, candidate)    #为每个用户训练一个Markov模型,得到每个用户的预测准确率,将准确率对用户取平均
    metrics['markov_acc'] = users_acc_markov

    if 'long' in parameters.model_mode:
        long_history = True
    else:
        long_history = False

    if long_history is False:
        data_train, train_idx = generate_input_history(parameters.data_neural, 'train', mode2=parameters.history_mode,
                                                       candidate=candidate)
        data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
                                                     candidate=candidate)
    elif long_history is True:
        if parameters.model_mode == 'simple_long':
            data_train, train_idx = generate_input_long_history2(parameters.data_neural, 'train', candidate=candidate)  #用于deepmove模型的训练数据?
            data_test, test_idx = generate_input_long_history2(parameters.data_neural, 'test', candidate=candidate)
        else:   #add cluster feature
            data_train, train_idx = generate_input_long_history(parameters.data_neural,parameters.vid_cluster, 'train', candidate=candidate)   #get the train data and the train index of sessions of each user
            data_test, test_idx = generate_input_long_history(parameters.data_neural,parameters.vid_cluster, 'test', candidate=candidate)  #get the test

    print('users:{} markov:{} train:{} test:{}'.format(len(candidate),avg_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))    #得到train data和test data的各自的样本数量
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    if not os.path.exists(SAVE_PATH + tmp_path):
        os.makedirs(SAVE_PATH + tmp_path,exist_ok=True)   #mkdir to makedirs
    print('*'*20+'start epoches'+'*'*20)
    for epoch in range(parameters.epoch):
        st = time.time()    #train start time
        if args.pretrain == 0:
            model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,criterion, parameters.model_mode,parameters.use_cuda)  #注意每个参数的意义
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            metrics['train_loss'].append(avg_loss)

        avg_loss, avg_acc, users_acc = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode)
        print('==>Test Acc:{} Loss:{:.4f}'.format(avg_acc, avg_loss))   #note avg_acc is a dict contains top1 fo top100 acc

        metrics['valid_loss'].append(avg_loss)  #each epoch's metric maybe loss should be adapt to topk loss
        metrics['accuracy'].append(avg_acc['top1'])     #select top5 as the inspect accuracy
        metrics['valid_acc'][epoch] = users_acc #valid_acc is 20 epoch, for each epoch list the acc of each user

        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        scheduler.step(float(avg_acc['top1']))
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:    #if learning rate decay
            load_epoch = np.argmax(metrics['accuracy']) #save the model in epoch that get the best accuracy in previous epochs
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp)) #load the best model parameters so far
            print('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:    #when lr is too low then break the training
            break
        if args.pretrain == 1:
            break

    mid = np.argmax(metrics['accuracy'])
    avg_acc = metrics['accuracy'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp)) #load the best model parameters so far
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)    #indent 是缩进的空格数, res.rs 保存训练所用参数和训练过程指标变化
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)  #txt show the train_loss,valid_loss,accuracy
    torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')    #save the best model to res.m

    for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):   #delete checkpoits
        for name in files:
            remove_path = os.path.join(rt, name)
            os.remove(remove_path)
    os.rmdir(SAVE_PATH + tmp_path)

    return avg_acc


def load_pretrained_model(config):
    res = json.load(open("./pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.pretrain = 0   #1


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--cluster_emb_size', type=int, default=10, help="cluster embeddings size")  #add cluster
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU', 'RNN'])  #select GRU
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])   #what is attention type dot is just xTy,general is xTWy 
    parser.add_argument('--data_path', type=str, default='../data/foursquare/')
    parser.add_argument('--save_path', type=str, default='../results/')
    parser.add_argument('--model_mode', type=str, default='attn_avg_long_user',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long']) #select model with avg attention
    parser.add_argument('--use_cuda',type=bool,default=False)
    parser.add_argument('--pretrain', type=int, default=0)  #original pretrain is set to 1
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)

    print('the final top1 accracy averaged on user is:',ours_acc)