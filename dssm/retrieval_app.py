#!/usr/local/bin/python
#-*- encoding:utf-8 -*-
import sys,os
import json

from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser

from jieba.analyse import ChineseAnalyzer

analyzer = ChineseAnalyzer()

schema = Schema(appid=ID(stored=True), 
                name=TEXT(stored=True, analyzer=analyzer), 
                description=TEXT(stored=True, analyzer=analyzer))
if not os.path.exists("index"):
    os.mkdir("index")

ix = open_dir("index") # for read only
 
with ix.searcher() as searcher:
    qp = qparser.QueryParser("content",ix.schema,group=qparser.syntax.OrGroup)
    c = searcher.collector(limit=10)
    tlc = TimeLimitCollector(c,timelimit=15)
    q = qp.parse(u'test # $')
    for pair in q.all_terms():
        print pair
    results = searcher.search_with_collector(q,tlc)
    
    if results.has_matched_terms():
        print 'YY', results.matched_terms()
    if 0!=len(results):
        for hit in results:
            print 'xx', hit['name'].encode('utf8')
    
            
            
            


