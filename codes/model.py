# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from haversine import haversine
import numpy as np

# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)    #hidden_size transform to loc_size,得到在locations上的分布吗
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name) #the name is model's name?
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name) #研究初始化细节
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)  #初始化ih，因为时object所有t是指向ih的引用，所以可以修改t从而修改ih
        for t in hh:
            nn.init.orthogonal_(t)  #初始化hh
        for t in b:
            nn.init.constant(t, 0)  #将b置0

    def forward(self, loc, tim):    #not user uid preference
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))  #为什么shape设为(1,1,hidden_size) because loc_emb and tim_emb shape is (steps,1,features)
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))  #initial hidden state and initial cell state (num_layers * num_directions, batch, hidden_size) num_layers=1,direction=1,batch=1
        if self.use_cuda:
            h1 = h1.cuda()  #delete cuda
            c1 = c1.cuda()  #delete cuda

        loc_emb = self.emb_loc(loc) #single loc is represented by a scalar value but into embedding autoly make it to onehot? (,1) to (,1,loc_embedsize).why not (,embdsize) 
        tim_emb = self.emb_tim(tim) #(,1) to (,1,tim_embedsize)
        x = torch.cat((loc_emb, tim_emb), 2)    #to (,1,loc+tim)
        x = self.dropout(x) #what does the dropout means? make some features of 2 dim not to join caculate

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))   #out is (steps,batch,hidden_size)
        out = out.squeeze(1)    #Returns a tensor with all the dimensions of input of size 1 removed. but here specify the dim=1,that is say if dim1's size is 1 then remove it else unchanged
        out = F.selu(out)   #relu的变种,在x<0时取一个较小的负值而不是0
        out = self.dropout(out) #为什么又用dropout,随机将tensor中的值置0来计算

        y = self.fc(out)    #shape is (steps,loc_size) 即每一步输出的是在所有候选地点上的概率分布(尚未归一化)
        score = F.log_softmax(y,dim=None)  # calculate loss by NLLoss.  equal to log(softmax(x)) but more stable output the same shape as input
        return score


# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size,use_cuda=False):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda=use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        if self.use_cuda:
            attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()    #delete cuda, it is the relation matrix between current state and history
        else:
            attn_energies = Variable(torch.zeros(state_len, seq_len))    #delete cuda, it is the relation matrix between current state and history

        for i in range(state_len):  #for current session's records' each hidden state,caculate the correlation between state and each history record 
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j])
        return F.softmax(attn_energies,dim=-1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output) #dot multiply is vector inner product
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy

# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.cluster_size= parameters.cluster_size  #add the cluster feature
        self.cluster_emb_size = parameters.cluster_emb_size #add the cluster feature
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
        self.emb_cluster = nn.Embedding(self.cluster_size,self.cluster_emb_size) #add the cluster feature
        input_size = self.loc_emb_size + self.tim_emb_size + self.cluster_emb_size

        # self.attn_d= Attn_d('distance',parameters.vid_lookup)   # add a new attn module
        self.attn = Attn(self.attn_type, self.hidden_size,self.use_cuda)    #
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)  #note 1 is the num_layers
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)  #
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim,cluster, history_loc,history_cluster, history_tim, history_count, uid, target_len):  #add cluster feature
        h1 = Variable(torch.zeros(1, 1, self.hidden_size),requires_grad=True)
        c1 = Variable(torch.zeros(1, 1, self.hidden_size),requires_grad=True)
        if self.use_cuda:
            h1 = h1.cuda()  #delete cuda
            c1 = c1.cuda()  #delete cuda
        ### use attn_d module to process the loc and history_loc
        # distr_d = self.attn_d(loc,history_loc)  # get a distribution on locations,add it to final out to tune the distribution
        ###
  
        loc_emb = self.emb_loc(loc) #note loc tim have the history records,add cuda
        cluster_emb = self.emb_cluster(cluster)    #add cluster feature
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, cluster_emb,tim_emb), 2) #add cluster feature 
        x = self.dropout(x)

        loc_emb_history = self.emb_loc(history_loc).squeeze(1)
        cluster_emb_history = self.emb_cluster(history_cluster).squeeze(1) #add cluster feature
        tim_emb_history = self.emb_tim(history_tim).squeeze(1)

        ### use time merge
        count = 0
        if self.use_cuda:
            loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1])).cuda() #delete cuda. what is the use of history_count? use it to merge the very timeclose records
            tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1])).cuda() #delete cuda
            cluster_emb_history2 = Variable(torch.zeros(len(history_count), cluster_emb_history.size()[-1])).cuda() #delete cuda , add cluster feature
        else:
            loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1])) #delete cuda. what is the use of history_count? use it to merge the very timeclose records
            tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1])) #delete cuda
            cluster_emb_history2 = Variable(torch.zeros(len(history_count), cluster_emb_history.size()[-1])) #delete cuda , add cluster feature


        for i, c in enumerate(history_count):
            if c == 1:
                tmp = loc_emb_history[count].unsqueeze(0)   #add a dim in place 0
                tmp_c = cluster_emb_history[count].unsqueeze(0) #add cluster feature
            else:
                tmp = torch.mean(loc_emb_history[count:count + c, :], dim=0, keepdim=True)  #merge the same time consecutive records,average the loc and time is the same
                tmp_c = torch.mean(cluster_emb_history[count:count + c, :], dim=0, keepdim=True) #add cluster feature
            loc_emb_history2[i, :] = tmp
            cluster_emb_history2[i, :] = tmp_c
            tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0)
            count += c  #because count:count+c 's records are merged,next count is count+c
        ###

        history = torch.cat((loc_emb_history2,cluster_emb_history2, tim_emb_history2), 1)    #concate (steps,520)
        history = F.tanh(self.fc_attn(history)) #trans to (steps,500)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out_state, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out_state, (h1, c1) = self.rnn(x, (h1, c1))
        out_state = out_state.squeeze(1)    #if dim 1's dim_value is 1 then remove it to get (steps,hidden_state)
        # out_state = F.selu(out_state)

        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0) #target_len is the length same as target,it is correspoding to current session. merge the history info related to the corresponding out_state into context,so context is (states,hiddensize) with same shape as selected out_state,e.g. out_state[-target_len:]
        context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0) #(1,states,history)*(history,hiddensize)=>(states,hiddensize),get the weight avg of history records for each state
        out = torch.cat((out_state[-target_len:], context), 1)  # no need for fc_attn   . get the (steps,out_state+context)

        uid_emb = self.emb_uid(uid).repeat(target_len, 1)   #add user preference in the end 
        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        ## use attn_d
        # y_final = y + torch.Tensor(np.repeat(distr_d,repeats=len(y),axis=0))
        # score = F.log_softmax(y_final)    #the log value of probability distribution on locations

        score = F.log_softmax(y,dim=-1)    #the log value of probability distribution on locations
        return score


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.uid_size=parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.cluster_size= parameters.cluster_size  #add the cluster feature
        self.cluster_emb_size = parameters.cluster_emb_size #add the cluster feature
        
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)   #uid add in the end and not through the rnn

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_cluster = nn.Embedding(self.cluster_size,self.cluster_emb_size) #add the cluster feature

        input_size = self.loc_emb_size + self.tim_emb_size + self.cluster_emb_size  #add cluster feature
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)  #what is encoder and decoder
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)  #encoder is for history,decoder is for current session
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)  #add uid feature
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self): #don't know
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim,cluster,uid, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        h2 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c2 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()  #delete cuda
            h2 = h2.cuda()  #delete cuda
            c1 = c1.cuda()  #delete cuda
            c2 = c2.cuda()  #delete cuda

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        cluster_emb = self.emb_cluster(cluster)    #add cluster feature

        x = torch.cat((loc_emb, tim_emb,cluster_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)  #note the rnn return sequence
            hidden_state, h2 = self.rnn_decoder(x[-target_len:], h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(x[-target_len:], (h2, c2))

        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        attn_weights = self.attn(hidden_state, hidden_history).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        uid_emb = self.emb_uid(uid).repeat(target_len, 1)   #add user preference in the end 
        out = torch.cat((hidden_state, context,uid_emb), 1)  #add user feature
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score
