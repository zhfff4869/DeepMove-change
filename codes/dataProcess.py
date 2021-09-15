from __future__ import print_function
from __future__ import division

import time
import argparse
import numpy as np
import pickle
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

def kdistgraph(path="../data/foursquare/locations.pk",k=100):
    with open(path,'rb') as file:
        data=pickle.load(file)
    dists=np.zeros(len(data))
    mink=np.zeros(len(data))
    for i in range(len(data)):
        point=data[i]
        for j in range(len(data)):
            if i==j: continue
            point1=data[j]
            mink[j]=np.sqrt(sum(np.power(point-point1,2)))
        mink=np.sort(mink)
        dists[i]=mink[k]
    dists=np.sort(dists)
    fig=plt.figure(figsize=(15,10))
    plt.plot(dists)
    plt.xlabel("venue")
    plt.ylabel("k-dist")
    plt.title("k-dist-graph")
    plt.save('../data/foursquare/k-dist-graph-nyc.png')
    plt.show()

def kdistgraph_plt(path='./dists.pk',save_path='../data/foursquare/'):
    with open(path,'rb') as file:
        dists=pickle.load(file)
    plt.figure(figsize=(15,10))
    plt.plot(dists)
    plt.xlabel("venue")
    plt.ylabel("k-dist")
    plt.title("k-dist-graph")
    plt.ylim(0,4)
    plt.savefig('../data/foursquare/k-dist-graph-nyc.png')
    plt.show()


class DataSet(object):
    def __init__(self, trace_min=10, split_gap=48,  session_min=2, session_record_max=10,
                 sessions_min=2, train_split=0.8, embedding_len=50):
        tmp_path = "../data/"
        self.dataPath="datacheck_1.csv"
        self.SAVE_PATH = tmp_path + 'foursquare/'   #changed
        self.save_name = 'foursquare_nyc'

        self.trace_len_min = trace_min  #轨迹最短长度
        # self.location_global_visit_min = global_visit   #
        self.split_gap = split_gap  #超过72小时分割session
        self.session_record_max = session_record_max
        self.filter_short_session = session_min  # not default 2, rather set to 5. session which less than 5 records was removed
        self.sessions_count_min = sessions_min
        self.train_split = train_split

        self.user_records=defaultdict(list)
        self.users={}
        self.venues={}
        self.activities={}
        self.clusters=defaultdict(int)
        self.venue_gps={}
        self.venue_activity={}
        self.venue_details_encode={}
        self.venue_cluster={}


        self.user_sessions={}
        self.user_sessions_encode={}

    def load_records(self):
        print("读取原数据")
        data=pd.read_csv(self.dataPath)
        for index,row in data.iterrows():
            user,venue,activity,lat,lon,tim,day,daytime=row['userId'],row['venueId'],row['venueCategory'],row['latitude'],row['longitude'],row['localtime'],row['day'],row['daytimelabel']
            try:
                tim=int(time.mktime(time.strptime(tim[:-6],"%Y-%m-%d %H:%M:%S")))
            except:
                print("该条记录处理失败，舍弃")
                continue
            self.user_records[user].append([user,venue,activity,lat,lon,tim,day,daytime])  #day是0-6 daytime是将一天划分为8段0-7


    def split_sessions(self):
        # 移除记录数太少的用户
        print("移除记录数太少的用户")
        user_remove=[]
        for user in self.user_records.keys():
            if len(self.user_records[user])<self.trace_len_min:
                user_remove.append(user)
        for user in user_remove:
            self.user_records.pop(user)
        # 切分sessions
        print("切分sessions")
        for user in self.user_records.keys():
            preTime = 0
            for i,record in enumerate(self.user_records[user]):
                if i==0 or len(self.user_sessions[user])==0:
                    self.user_sessions[user]=[[record]]
                    preTime=record[-3]
                    continue

                curTime=record[-3]
                deltaTime=curTime-preTime
                # 两个记录间隔超过48小时则切分，前一个session长度太长也切分
                if deltaTime/3600>self.split_gap or len(self.user_sessions[user][-1])>self.session_record_max:
                    # session的长度太短 视作噪声
                    if len(self.user_sessions[user][-1])<self.filter_short_session:
                        self.user_sessions[user].pop()
                    self.user_sessions[user].append([record])
                else:
                    self.user_sessions[user][-1].append(record)

                preTime = curTime
            #  用户的最后一个session如果太短要去掉
            if len(self.user_sessions[user][-1]) < self.filter_short_session:
                self.user_sessions[user].pop()

        # 移除session数量过少的用户
        print("移除session数量过少的用户")
        user_remove2=[]
        for user in self.user_sessions:
            if len(self.user_sessions[user])<self.sessions_count_min:
                user_remove2.append(user)
        for user in user_remove2:
            self.user_sessions.pop(user)

        # 剩余的用户集，地点集，活动集 以及编码字典
        print("构建剩余的用户集，地点集，活动集 以及编码字典")
        for user in self.user_sessions:
            if user not in self.users:
                self.users[user] = len(self.users)
            for session in self.user_sessions[user]:
                for record in session:
                    venue,activity,lat,lon=record[1:5]
                    if venue not in self.venues:
                        self.venues[venue]=len(self.venues)
                        self.venue_gps[venue]=[lat,lon]
                        self.venue_activity[venue]=activity

                    if activity not in self.activities:
                        self.activities[activity]=len(self.activities)
        # 构建venue_details_encode
        for venue in self.venues:
            self.venue_details_encode[venue]=[self.venue_gps[venue][0],self.venue_gps[venue][1],self.activities[self.venue_activity[venue]]]
        print('DBSCAN对venue, activity, lat, lon聚类得到vid_cluster')
        # DBSCAN对venue, activity, lat, lon聚类得到vid_cluster
        locations=list(self.venue_details_encode.values())
        locations=np.array(locations)
        db=DBSCAN(eps=0.5,min_samples=100).fit(locations)
        clusters=db.labels_+1
        for i,venue in enumerate(self.venues.keys()):
            self.venue_cluster[venue]=clusters[i]
        print('构建cluster集')
        # 构建cluster集
        for i in clusters:
            self.clusters[i]+=1
        # 计算k-dist-graph
        # pickle.dump(locations,open("../data/foursquare/locations.pk",'wb'))

        # 对sessions进行编码
        print("对sessions进行编码")
        for user in self.user_sessions:
            self.user_sessions_encode[user]=[]
            for session in self.user_sessions[user]:
                self.user_sessions_encode[user].append([])
                for record in session:
                    _, venue, activity, lat, lon, tim, day, daytime=record
                    cluster = self.venue_cluster[venue] #先将与venue相关的所有属性编码 最后将venue编码
                    venue=self.venues[venue]
                    activity=self.activities[activity]
                    self.user_sessions_encode[user][-1].append([venue, activity, cluster, tim, day, daytime])

        user_sessions_encode2={}
        for user,sessions in self.user_sessions_encode.items():
            uid = self.users[user]
            user_sessions_encode2[uid]=sessions
        self.user_sessions_encode=user_sessions_encode2


    def save(self):
        print("保存处理完毕的数据到：",self.SAVE_PATH+self.save_name+'.pk')
        foursquare_nyc={'user_sessions_encode':self.user_sessions_encode,
                        'user_uid':self.users,'venue_vid':self.venues,'activity_aid':self.activities,'cluster_visit':self.clusters,
                        'venue_activity':self.venue_activity,'venue_cluster':self.venue_cluster,'venue_gps':self.venue_gps,
                        'user_size':len(self.users),
                        'venue_size':len(self.venues),
                        'activity_size':len(self.activities),
                        'cluster_size':len(self.clusters),
                        'day_size':7,
                        'daytime_size':8}
        pickle.dump(foursquare_nyc,open(self.SAVE_PATH+self.save_name+'.pk','wb'))


if __name__ == '__main__':
    # kdistgraph()
    # kdistgraph_plt()
    dataset=DataSet()
    dataset.load_records()
    dataset.split_sessions()
    dataset.save()
