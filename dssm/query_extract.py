# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:15:48 2014

@author: jjhu
"""

import jieba
import sys
import pickle
from w2v_embed_record import WordEmbedding

def main():
    we = pickle.load(open('w2v_embed.pkl','r'))
    print "----test----"
    print we.word2id['UNKNOWN']
    
    query_info = {}
    cnt = 0
    for line in sys.stdin:
        li = line.strip().split('\t',1)
        if len(li)!=6:
            print>>sys.stderr, 'wrong line:', line
            continue
        try:
            print li[0]
            query_list = jieba.cut(li[0],cut_all=False)
            query_info[cnt]={}
            tmp_list=query_list.strip().split()
            query_info[cnt]['query']=tmp_list
            query_info[cnt]['query2id']=[]
            for item in tmp_list:
                if item in we.word2id.keys():
                    query_info[cnt]['query2id'].append(we.word2id[item])
                else:
                    print "query word not in dictionary", item
                    query_info[cnt]['query2id'].append(we.word2id[item])
                    
            query_info[cnt]['search']=int(line[1])
            query_info[cnt]['appoid']=line[2]
            query_info[cnt]['display']=int(line[3])
            query_info[cnt]['click']=int(line[4])
            query_info[cnt]['download']=int(line[5])
        except Exception as e:
            pass
    with open('query_info.pkl','w') as f:
        pickle.dump(query_info,f)
        
if __name__ == '__main__':
    main()
                    
                    
                    
        