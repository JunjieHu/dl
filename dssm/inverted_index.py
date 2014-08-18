#!/usr/local/bin/python
#-*- encoding:utf-8 -*-
import sys,os
import json

from whoosh.index import create_in
from whoosh.fields import *
#from whoosh.qparser import QueryParser

from jieba.analyse import ChineseAnalyzer

analyzer = ChineseAnalyzer()

schema = Schema(appid=ID(stored=True), 
                name=TEXT(stored=True, analyzer=analyzer), 
                description=TEXT(stored=True, analyzer=analyzer))
if not os.path.exists("index"):
    os.mkdir("index")

ix = create_in("index", schema) # for create new index
#ix = open_dir("tmp") # for read only
writer = ix.writer()

cnt = 0
for line in sys.stdin:
    li = line.strip().split('\t')
    if len(li)!=2:
        print>>sys.stderr, 'wrong', line
        continue
    print li[0]
    try:
        res = json.loads(li[1].decode('utf8'))
        if 'detailInfo' in res and len(res['detailInfo'])>0:
            print cnt 
            cnt = cnt+1
            detailInfo = res['detailInfo'][0]
            name = unicode(detailInfo['name']).strip()
            description = unicode(detailInfo['description']).strip()
    
            # write to ix.writer
            writer.add_document(appid=li[0].decode('utf8'),
                                name=name.decode('utf8'),
                                description=description.decode('utf8'))            
            
    except Exception as e:
        pass
    if cnt > 10:
        print cnt
        break
writer.commit()    
            
            
            


