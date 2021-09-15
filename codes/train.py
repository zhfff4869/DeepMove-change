# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable

import numpy as np
import pickle
from collections import deque, Counter


class RnnParameterData(object):
    def __init__(
        self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50,
        tim_emb_size=10, cluster_emb_size=10, hidden_size=500,
        lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, 
        clip=5.0, optim='Adam',history_mode='avg', attn_type='dot',
        epoch_max=30, rnn_type='LSTM', model_mode="simple",
        data_path='../data/', save_path='../results/', data_name='foursquare',use_cuda=True
        ):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        with open(self.data_path + self.data_name + '.pk', 'rb') as f:
            data = pickle.load(f,encoding='latin1') #decoding error, use latin1 to encoding
        self.vid_list = data['vid_list']    #vid_list struct is pid:(vid,visit_times).vid is pid's code from 0 to len(pids). 0 is the unknown loc
        self.uid_list = data['uid_list']    #uid_list struct is user:(uid,sessions_length),uid is the code from 0 to len(users)-1
        self.data_neural = data['data_neural']
        self.vid_lookup=data['vid_lookup']  #add the coordinate
        self.vid_cluster=data['vid_cluster']    #add the cluster feature

        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.cluster_size= self.vid_cluster['num_clusters'] #add the cluster feature

        self.loc_emb_size = loc_emb_size    #location embedding
        self.tim_emb_size = tim_emb_size    #time embedding
        self.voc_emb_size = voc_emb_size    #voc is what?still not know?
        self.uid_emb_size = uid_emb_size    #user embedding
        self.cluster_emb_size= cluster_emb_size  #add the cluster feature
        self.hidden_size = hidden_size      #RNN's hidden size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = use_cuda #True
        self.lr = lr
        self.lr_step = lr_step  #learning step? how many step decay learning rate?still question?
        self.lr_decay = lr_decay    #decay lr's parameter
        self.optim = optim      #optimizer's type
        self.L2 = L2    #L2 normalization's lambda?
        self.clip = clip    #what is clip? caculate grad and sum all grad's value and it have to less than clip

        self.attn_type = attn_type  #what is attention type dot? matrix dot multiply, look it up in Deepmove2020
        self.rnn_type = rnn_type
        self.history_mode = history_mode    #note history mode
        self.model_mode = model_mode    #simple long is ?


def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            # voc_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 27))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            # trace['voc'] = Variable(torch.LongTensor(voc_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            if mode2 == 'avg':
                trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_input_long_history2(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate: #注意candidate is users. each user's records split to train trace and test trace
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}  #记录了每个用户的训练数据

        trace = {}      #
        session = []    #
        for c, i in enumerate(train_id):
            session.extend(sessions[i]) #concate these sessions in the session variable 注意只有选了sessions中的前面一些的session用于训练
        target = np.array([s[0] for s in session[1:]])  #向后错开一位得到target位置序列,长度比session小1

        loc_tim = []    #location and time
        loc_tim.extend([(s[0], s[1]) for s in session[:-1]])    #去掉最后一个得到与target对应的输入序列,注意输入序列包含位置和时间2个特征
        loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))   #loc 输入序列 reshape to (-1,1)
        tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))   #time 输入序列
        trace['loc'] = Variable(torch.LongTensor(loc_np))   #将numpy转为longtensor类型,再用Variable包裹起来,Variable是使其能够自动计算梯度吗?
        trace['tim'] = Variable(torch.LongTensor(tim_np))
        trace['target'] = Variable(torch.LongTensor(target))    #target只有1个维度
        data_train[u][i] = trace    #trace是u的完整的训练记录序列,i是指的这个记录序列包含的session在sessions中的最大index
        # train_idx[u] = train_id
        if mode == 'train':
            train_idx[u] = [0, i]   #sessions中被截取为训练集的session's index 从0到i
        else:
            train_idx[u] = [i]  #sessions中被截取为测试集的最后一个session index
    return data_train, train_idx    


def generate_input_long_history(data_neural,vid_cluster, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:  #ignore train_data's first one session, because it will be no history
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])  #对于当前session，除了第一个之外后面每一步的loc都要作为目标地点参与训练

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']   #在test时先加上train中的数据作为历史
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):  #c是当前的session下标,将c之前从下标0开始的session中的record作为历史，注意当前session的record没有加入history
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])   

            history_tim = [t[1] for t in history]   #history time sequence
            history_count = [1]     #what is this?for timeslot loc merge
            last_t = history_tim[0] #is this the initial time? why call it last time. because below relate to t it is last time
            count = 1   #what does the count means
            for t in history_tim[1:]:
                if t == last_t: #if the time is same as last time,then means the 2 records are very time close?or means repeatetion?
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1) #what is history count? history_tim is like [1,2,2,3,3,4],then history_count is [1,2,2,1] means 1 has once,then 2 has twice,then 3 has twice,then 4 has once
                    last_t = t
                    count = 1

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_cluster = np.reshape(np.array([vid_cluster[s[0]] for s in history]), (len(history), 1)) #add cluster feature
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_cluster'] = Variable(torch.LongTensor(history_cluster))  #add cluster feature
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count  #将时间序列中重复的时间合并来存储,不知道有什么用

            loc_tim = history   #将历史和当前session连起来， the loc_tim contains history+current，最后一条record除外，因为没有对应的target
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])    #history extend the current session except the last record
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            cluster_np = np.reshape(np.array([vid_cluster[s[0]] for s in loc_tim]), (len(loc_tim), 1))  #add cluster feature
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))   #loc_np into trace['loc']
            trace['cluster'] = Variable(torch.LongTensor(cluster_np)) #add cluster feature
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))    #note target is the [1:] of current session
            data_train[u][i] = trace    #train_id is training trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys())   #py2 error
    train_queue = deque()
    if mode == 'random':    #为啥不直接shuffle就完了,慢慢地pop也太麻烦了吧 考虑修改代码.note this is designed for final model rather than simple_long
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])  #why ignore index 0?because 0 index means there is no history. 对于simple_long模型：training deque要用iterable来初始化,所以用[1:]得到一个序列,这里序列中只有1个值. 
            else:
                initial_queue[u] = deque(train_idx[u])      #for final model:test_idx's history have the train_idx so not need ignore 0 .for simple_long:test_idx 由于本来就是(,1)的数据,所以直接用
        queue_left = 1  #flag that represent if how many users still have traces that not poped
        while queue_left > 0:   #当initial_queue还没pop干净则要继续pop
            np.random.shuffle(user) #'dict_keys' object is not subscriptable, user=> list(user) 每次都要重新打乱user顺序吗，这样不是会重复吗
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:   #首先检查是否是已经pop掉值的用户,避免重复pop导致empty deque pop error
                    train_queue.append((u, initial_queue[u].popleft())) #value=(user,train_idx[-1]),注意initial_queue[u]的值pop掉了
                if j >= int(0.01 * len(user)):  #train_queue中每次pop into百分之一的user的train_idx数据?one time pop some index into
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])   #filter null user,whose value was poped to train_queue,then count user number to queue_left
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))  #append (user,session_index)
    return train_queue


def get_acc(target, scores):    #可以考虑加一个top20,top50,毕竟地点数量有上万个
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(100, 1) #return 100 largest value and corresponding index in score along dim 1 .can change it to top50
    predx = idxx.cpu().numpy()  #this is top10 prediction loc id for each step
    acc = np.zeros((6, 1))  #dim0 is top10 acc, dim1 is top5 acc, dim2 is top1 acc
    for i, p in enumerate(predx):
        t = target[i]   #t is the true loc in i step
        if t in p[:100] and t > 0:
            acc[5] += 1
        if t in p[:50] and t > 0:
            acc[4] += 1
        if t in p[:20] and t > 0:
            acc[3] += 1
        if t in p[:10] and t > 0:   #if t is in top10 predict loc. note 0 is the unknown loc
            acc[2] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[0] += 1

    return acc


def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count


def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None,use_cuda=False): #default train/test processing 
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        model.train(True)   #将模型设为训练模式,这是pytorch的Module模块的自带函数
        run_queue = generate_queue(run_idx, 'random', 'train')  #return deque contains (user,train_idx)
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)  #因为是每个用户的数据划分为训练集和测试集,所以训练集和测试集的用户数量是相同的

    users_acc = {}  #accuracy of each user
    for c in range(queue_len):
        optimizer.zero_grad()   #将各参数梯度初始化为0
        u, i = run_queue.popleft()  #for final model:here i is the idx of a trace of the user   .for simple_long:note run_queue length is decreasing. i is the last idx of train indexs,这里提取了用于训练的一个样本的指标信息
        if u not in users_acc:  
            users_acc[u] = [0, 0,0,0,0,0,0]   #0 dim is predict_count,then is top1 top5 top10 top20 top50 top100

        if use_cuda:
            loc = data[u][i]['loc'].cuda()  #cuda deleted,get the history concate the current? .   for simple_long:这里只选了train data地最后一个idx?之前处理训练数据的时候就已经将数据整合然后设key=train_idx的最后一个idx了
            cluster = data[u][i]['cluster'].cuda() #add cluster feature
            tim = data[u][i]['tim'].cuda()  #cuda deleted,     .所以[i]就是直接取整合好的训练数据
            target = data[u][i]['target'].cuda()    #cuda deleted
            uid = Variable(torch.LongTensor([u])).cuda()    #cuda deleted
        else:
            loc = data[u][i]['loc']  #cuda deleted,get the history concate the current? .   for simple_long:这里只选了train data地最后一个idx?之前处理训练数据的时候就已经将数据整合然后设key=train_idx的最后一个idx了
            cluster = data[u][i]['cluster'] #add cluster feature
            tim = data[u][i]['tim']  #cuda deleted,     .所以[i]就是直接取整合好的训练数据
            target = data[u][i]['target']   #cuda deleted
            uid = Variable(torch.LongTensor([u]))

        if 'attn' in mode2:
            if use_cuda:
                history_loc = data[u][i]['history_loc'].cuda()  #cuda deleted. get the history loc
                history_cluster = data[u][i]['history_cluster'].cuda() #add cluster feature
                history_tim = data[u][i]['history_tim'].cuda()  #cuda deleted. get the history time
            else:
                history_loc = data[u][i]['history_loc']  #cuda deleted. get the history loc
                history_cluster = data[u][i]['history_cluster'] #add cluster feature
                history_tim = data[u][i]['history_tim']

        if mode2 in ['simple', 'simple_long']:
            scores = model(loc, tim)
        elif mode2 == 'attn_avg_long_user':
            history_count = data[u][i]['history_count'] #note how to use history_count,it is used to merge the same time records
            target_len = target.data.size()[0]  #等于当前session中记录长度减1
            scores = model(loc, tim,cluster, history_loc,history_cluster, history_tim, history_count, uid, target_len)
        elif mode2 == 'attn_local_long':    #this is final model 
            target_len = target.data.size()[0]
            scores = model(loc, tim,cluster,uid, target_len)

        if scores.data.size()[0] > target.data.size()[0]:   #adapt size of output to target, disgard the start sevaral steps' prediction
            scores = scores[-target.data.size()[0]:]
        #可以考虑加入weight使得最初几步的预测权重小,权重随序列长度逐渐增大  for criterion, input=torch.tensor([[0.1,0.2,0.7],[0.4,0.5,0.1]]),target=torch.tensor([2,1]),output=1/2*0.7+1/2*0.5=0.6
        loss = criterion(scores, target)    #criterion autoly encode target to onehot? Input: (N,C) where C = number of classes,target:(N) where each value is 0≤targets[i]≤C−1
        if mode == 'train': #只有训练才反向传播更新参数，测试则直接forward就行
            loss.backward() #反向传播用更新梯度
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    #scale使得total_norm<clip
                for p in model.parameters():    #这3步似乎不需要
                    if p.requires_grad:
                        p.data.add_(p.grad.data,-lr)   #为什么用梯度减lr?不是data=data-lr*grad吗 逗号是不是该改成乘法?这一步似乎只是对clip函数的修饰?
            except:
                pass
            optimizer.step()    #这一步才是使用梯度来更新每个参数
        elif mode == 'test':   
            users_acc[u][0] += len(target)  #dim0 is target length,which means predict step count
            acc = get_acc(target, scores)
            users_acc[u][1] += acc[0]   #dim1 get top1 acc count ,保存top5 top10 top20 top50 top100
            users_acc[u][2] += acc[1]
            users_acc[u][3] += acc[2]
            users_acc[u][4] += acc[3]
            users_acc[u][5] += acc[4]
            users_acc[u][6] += acc[5]
        total_loss.append(loss.data.cpu().numpy())  #delete [0]

    avg_loss = np.mean(total_loss, dtype=np.float64)   #consider change the loss function to topk loss? 
    if mode == 'train':
        return model, avg_loss  #trained model and average loss
    elif mode == 'test':
        users_rnn_acc = {}  #top1 acc of each user
        for u in users_acc: # get the users_acc only in test
            tmp_acc1 = users_acc[u][1] / users_acc[u][0]
            tmp_acc2 = users_acc[u][2] / users_acc[u][0]
            tmp_acc3 = users_acc[u][3] / users_acc[u][0]
            tmp_acc4 = users_acc[u][4] / users_acc[u][0]
            tmp_acc5 = users_acc[u][5] / users_acc[u][0]
            tmp_acc6 = users_acc[u][6] / users_acc[u][0]
            users_rnn_acc[u] = {
                'top1':tmp_acc1.tolist()[0],'top5':tmp_acc2.tolist()[0],
                'top10':tmp_acc3.tolist()[0],'top20':tmp_acc4.tolist()[0],
                'top50':tmp_acc5.tolist()[0],'top100':tmp_acc6.tolist()[0]}


        avg_acc1 = np.mean([users_rnn_acc[x]['top1'] for x in users_rnn_acc])    #top1 acc averaged on users
        avg_acc5 = np.mean([users_rnn_acc[x]['top5'] for x in users_rnn_acc])    #top1 acc averaged on users
        avg_acc10 = np.mean([users_rnn_acc[x]['top10'] for x in users_rnn_acc])    #top1 acc averaged on users
        avg_acc20 = np.mean([users_rnn_acc[x]['top20'] for x in users_rnn_acc])    #top1 acc averaged on users
        avg_acc50 = np.mean([users_rnn_acc[x]['top50'] for x in users_rnn_acc])    #top1 acc averaged on users
        avg_acc100 = np.mean([users_rnn_acc[x]['top100'] for x in users_rnn_acc])    #top1 acc averaged on users
        avg_acc={'top1':avg_acc1,'top5':avg_acc5,'top10':avg_acc10,'top20':avg_acc20,'top50':avg_acc50,'top100':avg_acc100}

        return avg_loss, avg_acc, users_rnn_acc


def markov(parameters, candidate):  #in markov we only use the transition of location
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions']  #注意spatiotemporal point q 是 (t,l) 还是 (l,t)? 看来是(l,t)
        train_id = parameters.data_neural[u]['train']   #train_id 是什么，traces中的那些用来train,
        test_id = parameters.data_neural[u]['test']     #哪些用来test
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])  #extract vid sessions by original structure
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)   #concate servaral sessions to one
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])   #similar to train_id
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]   #为什么赋值给了validation? save the train_data and test_data of each user
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys(): #对每个用户do
        topk = list(set(validation[u][0]))  #用户的训练数据的地点集合
        transfer = np.zeros((len(topk), len(topk))) #计算转移概率矩阵 每个用户都有属于自己的转移概率矩阵

        # train
        sessions = parameters.data_neural[u]['sessions']    #seems can use validation[u] for simplify
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):    #除去最后1个? train_data remove the last one,test_data remove the first one
                loc = s[0]
                target = sessions[i][j + 1][0]  #下一个地点就是target
                if loc in topk and target in topk:  #如果loc和target都在topk中,则获取各自的指标,将对应的转移次数加1
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):  #将每行的转移归一化为概率
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0 #每个用户的预测准确度
        test_id = parameters.data_neural[u]['test'] #与train类似
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):    #for a session
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1      #与acc对应
                user_count += 1 #与user_acc对应
                if loc in topk: #topk还是训练数据的地点集 如果这个地点没有在训练集中出现过则忽略
                    pred = np.argmax(transfer[topk.index(loc), :])  #预测下一个地点的index
                    if pred >= len(topk) - 1:   #预测出的地点在训练集中没有出现过就算无效
                        pred = np.random.randint(len(topk)) #预测的是最大的地点指标,则随机再topk里选一个?为什么,预测最大index的地点是无效的,因为计算转移概率时每个session都去掉了最后一条记录

                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1            #总的准确度，acc和count似乎没有用啊,最后还是按照user数量来平均
                        user_acc[u] += 1    #每个用户的预测准确度
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, user_acc
