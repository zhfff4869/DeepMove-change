from __future__ import print_function
from __future__ import division

import time
import argparse
import numpy as np
import pickle
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN


def entropy_spatial(sessions):
    locations = {}
    days = sorted(sessions.keys())  #get a user's sessions keys
    for d in days:  #count the user's location visits
        session = sessions[d]
        for s in session:   #
            if s[0] not in locations:
                locations[s[0]] = 1
            else:
                locations[s[0]] += 1
    frequency = np.array([locations[loc] for loc in locations])
    frequency = frequency / np.sum(frequency)
    entropy = - np.sum(frequency * np.log(frequency))   #caculate the entropy
    return entropy


class DataFoursquare(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=72, min_gap=10, session_min=2, session_max=10,
                 sessions_min=2, train_split=0.8, embedding_len=50):
        tmp_path = "../data/"
        self.TWITTER_PATH = tmp_path + 'foursquare/tweets_clean.txt'    #不确定是否格式符合
        self.VENUES_PATH = tmp_path + 'foursquare/venues_all.txt'       #这个是聚类得到的poi,坐标信息和地点类别?
        self.SAVE_PATH = tmp_path + 'foursquare/'   #changed
        self.save_name = 'foursquare'

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min #not default 2, rather set to 5. session which less than 5 records was removed
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.words_original = []    #这是文本信息?
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}    #what does unk means
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.vid_cluster={}
        self.pid_loc_lat = {}   #does this means coordinate information?
        self.data_neural = {}   #意思是处理好的数据 this is the final processed data

    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        with open(self.TWITTER_PATH,encoding='utf-8') as fid:   #change: append encoding=utf-8 rather than gbk
            for i, line in enumerate(fid):
                _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('') # 第一个是啥...  ,does pid means placeId?seems right
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]   #data structure is [location,time] 考虑加入地点类别和坐标
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:
                    self.venues[pid] = 1    #venues save the loc and corresponding count
                else:
                    self.venues[pid] += 1
    # ############# 1.read other dataset


    # ########### 3.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self):
        uid_3 = [x for x in self.data if len(self.data[x]) > self.trace_len_min]    #filter the users whose records length<trace_len_min
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)   #sort (uid,records_len) by records_len from max to min
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min] #filter the venues where was visited less than location_global_visit_min
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)   #sort (pid,visit_count) by visit_count from max to min
        pid_3 = dict(pid_pic3) #transform to dict struct pid:visit_count

        session_len_list = []
        for u in pick3: #u structure is (uid,records_len)
            uid = u[0]  #get the uid
            info = self.data[uid]   #get the (pid,time) records of uid
            topk = Counter([x[0] for x in info]).most_common()  #count visit times of each pid of the uid
            topk1 = [x[0] for x in topk if x[1] > 1]    #filter pid which is visited less than twice by the uid
            sessions = {}   #split trace to sessions
            for i, record in enumerate(info):   #still use original info? used next to judge whether ignore the record
                poi, tmd = record
                try:
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S"))) #get the timestamp in seconds
                except Exception as e:  #if the time read error then discard this record
                    print('error:{}'.format(e)) 
                    continue
                sid = len(sessions) #session Id? to represent the index of each session in dict sessions
                if poi not in pid_3 and poi not in topk1:   #if poi is not visited globally more than 10 times and is not visited locally more than 1,then ignore it
                    # if poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0:    #first record to first session, sessions length 0 condition is for when first record of the user is discard
                    sessions[sid] = [record]    #init first session of sessions
                else:
                    # if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:    #if surpass the timegap or the current sesssion's length more than sesssion_max,then start a new session
                    #     sessions[sid] = [record]
                    if (tid - last_tid) / 3600 > self.hour_gap:    #if surpass the timegap or the current sesssion's length more than sesssion_max,then start a new session
                        sessions[sid] = [record]
                    elif (tid - last_tid) / 60 > self.min_gap:  #if timedelta less than 10 min,then regard it as noise. generally its loc will be same as last record,or it will vary like noise,so ignore it
                        sessions[sid - 1].append(record)
                    else:
                        pass
                last_tid = tid  #remember the time for next record
            sessions_filter = {}    #get processed sessions
            for s in sessions:  #for each session, note sessions is of a user
                if len(sessions[s]) >= self.filter_short_session:   #get the valid session into sessions_filter
                    sessions_filter[len(sessions_filter)] = sessions[s] #note len(dict) is used for represent dict's index order
                    session_len_list.append(len(sessions[s]))   #corresponding length of each session
            if len(sessions_filter) >= self.sessions_count_min: #user whose length of sessions more than or equal to sessions_count_min is valid, else remove the user
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions} #note the topk is raw count but sessions_filter is processed,topk not used next
        #user_filter3 is the users of sessions_filter,and actually the judgement has overlaped,we remove it
        self.user_filter3 = [x for x in self.data_filter]  #here just save the uids whose sessions length more than session_count_min,but seems indeed overlap at filter condition

    # ########### 4. build dictionary for users and location
    def build_users_locations_dict(self):
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            if u not in self.uid_list:  #note here uid_list is for re-encode user to uid from 0
                self.uid_list[u] = [len(self.uid_list), len(sessions)]  #length of uid_list represent the index inited from 0  of uid in user_filter3             
            for sid in sessions:    #for each session
                poi = [p[0] for p in sessions[sid]] #get the poi sequence
                for p in poi:   #note vid_list re-encode the poi to vid from 1,while 0 is the unknown location
                    if p not in self.vid_list:  #note unknown loc's poi is 0. unk is unknown, the visited times set -1
                        self.vid_list_lookup[len(self.vid_list)] = p    #here encode all poi from 1 begin,as 0 is unknown poi
                        self.vid_list[p] = [len(self.vid_list), 1]  #dict from poi to (vid,visit_count)
                    else:
                        self.vid_list[p][1] += 1    #and count the poi visited times

    # support for radius of gyration
    def load_venues(self):
        with open(self.TWITTER_PATH, encoding='utf-8') as fid:  #change 'r' to encoding='utf-8'
            for line in fid:
                _, uid, lat, lon, tim, _, _, tweet, pid = line.strip('\r\n').split('') #latitude longitude写反了? first dim is lat second dim is lon,but it seems doesn't matter for using
                self.pid_loc_lat[pid] = [float(lat), float(lon)]    #get the coordinate

    def venues_lookup(self):
        for vid in self.vid_list_lookup:    #note the vid is encoded index form 1 of placeId. does vid means visitId?
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid] #here can add a vid_cluster dict
            self.vid_lookup[vid] = lon_lat
        coords=np.array(list(self.vid_lookup.values()))
        kms_rad=6371.088
        epsilon=0.5/kms_rad #this is transform km to radian on earth,radian=distance/radius,note haversine return km
        db= DBSCAN(eps=epsilon, min_samples=5, algorithm="ball_tree",metric="haversine").fit(np.radians(coords))
        cluster_labels=db.labels_   # for each point,label it with the cluster id
        cluster_labels_0=cluster_labels+1
        num_clusters=len(set(cluster_labels_0)) #-1 is the noise,or we say unknown,but we add 1 to all to set 0 as unknown
        for vid in self.vid_lookup:
            self.vid_cluster[vid]=cluster_labels_0[vid-1]
        self.vid_cluster['num_clusters']=num_clusters
        print("cluster number:",num_clusters)

        

    # ########## 5.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):   #time encoder
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        if tm.tm_wday in [0, 1, 2, 3, 4]:   #tm_wday range[0,6] Monday is 0. tm_hour range[0,23]
            tid = tm.tm_hour    #when it's workday,time is just the hour range[0,23]
        else:
            tid = tm.tm_hour + 24   #when it's weekend,time add 24 to represent weekend's hour. so if time>23,it is weekend time
        return tid

    def prepare_neural_data(self):  #this is for train directly
        for u in self.uid_list:
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}  #this is sessions for training
            sessions_id = []    #sessions each session's id
            for sid in sessions: #for each session
                sessions_tran[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]    #p is record in the session. encode each record's time to tid and poi to vid. struct is (vid,tid)
                sessions_id.append(sid)
            split_id = int(np.floor(self.train_split * len(sessions_id)))   
            train_id = sessions_id[:split_id]   #train session id
            test_id = sessions_id[split_id:]    #test session id
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])   #in train,each session ignore first one,the left records are for groudtruth,sum all to get the groudtruth num of the user
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])   #in test,get the groudtruth num of the user
            train_loc = {}  #user's loc visits in train
            for i in train_id:  #count each user's loc,visits in train
                for sess in sessions_tran[i]:   #sess is a record
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            # calculate entropy
            entropy = entropy_spatial(sessions) #entropy of the loc visits distribution of the user

            # calculate location ratio
            train_location = []
            for i in train_id:  #for each train session. here squeeze pid in sessions of the user to a list
                train_location.extend([s[0] for s in sessions[i]])  #s is record,s[0] is pid
            train_location_set = set(train_location)    #train location category
            test_location = []  #so as test
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set #location category of the user's all sessions
            test_unique = whole_location - train_location_set   #the loc unique in test but not in train
            explore = len(test_unique) / len(whole_location) #what is this's meaning this is explore

            # calculate radius of gyration
            lon_lat = []    #in fact the struct is (lat,lon)?
            for pid in self.pid_loc_lat.keys():
                try:
                    lon_lat.append(self.pid_loc_lat[pid])
                except:
                    print(pid)
                    print('error')
            lon_lat = np.array(lon_lat) #this is train coordinates
            center = np.mean(lon_lat, axis=0, keepdims=True)    #the center coordinate
            center = np.repeat(center, axis=0, repeats=len(lon_lat))    #repeat the center coordinate to same shape as lon_cat
            rg = np.sqrt(np.mean(np.sum((lon_lat - center) ** 2, axis=1, keepdims=True), axis=0))[0]    #average distance from coordinates to center

            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': explore,
                                                     'entropy': entropy, 'rg': rg}  #note user is encoded from 1 to len(users)

    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        parameters['TWITTER_PATH'] = self.TWITTER_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH

        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min    #统计所有用户的访问次数和小于10的地点要舍弃
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap    #what is min gap used for
        parameters['session_max'] = self.session_max    #session_max means session's length can't more than max? so weird
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min    #each user should have more than or equal to 5 sessions
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):   #note uid_list struct is (uid,sessions_number),vid_list is (vid,visit_times)
        foursquare_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup,'vid_cluster':self.vid_cluster
                              }    #vid_lookup is from vid to look its coordinate
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold")
    parser.add_argument('--hour_gap', type=int, default=72, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=10, help="minimum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=5, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                    hour_gap=args.hour_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, train_split=args.train_split)
    parameters = data_generator.get_parameters()    #note objects are ignored by get_parameters()
    print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    print('############START PROCESSING:')
    print('load trajectory from {}'.format(data_generator.TWITTER_PATH))
    data_generator.load_trajectory_from_tweets()
    print('filter users')
    data_generator.filter_users_by_length()
    print('build users/locations dictionary')
    data_generator.build_users_locations_dict() #here
    data_generator.load_venues()
    data_generator.venues_lookup()
    print('prepare data for neural network')
    data_generator.prepare_neural_data()
    print('save prepared data')
    data_generator.save_variables()
    print('raw users:{} raw locations:{}'.format(
        len(data_generator.data), len(data_generator.venues)))
    print('final users:{} final locations:{}'.format(
        len(data_generator.data_neural), len(data_generator.vid_list)))
