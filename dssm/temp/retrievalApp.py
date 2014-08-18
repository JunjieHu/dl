#!/usr/local/bin/python
#-*- encoding:utf-8 -*- 
 
from whoosh.index import open_dir  
from whoosh.fields import *  
from whoosh import qparser;
from chinesetokenizer import ChineseAnalyzer
#from whoosh.analysis import RegexAnalyzer  
#analyzer = RegexAnalyzer(ur"([\u4e00-\u9fa5])|(\w+(\.?\w+)*)")
from whoosh.collectors import TimeLimitCollector, TimeLimit
analyzer = ChineseAnalyzer()

ix = open_dir('IndexDir/titleIndex'); 


with ix.searcher() as searcher:
    qp = qparser.QueryParser("content", ix.schema,group=qparser.syntax.OrGroup);
    c = searcher.collector(limit=10);
    tlc = TimeLimitCollector(c, timelimit=15);
    q = qp.parse(u'五子棋GOMOKU')
    for pair in q.all_terms():
        print pair;
    results = searcher.search_with_collector(q, tlc);
    print 'Here'
    if results.has_matched_terms():
        print('YYY',results.matched_terms())
    if 0 != len(results):
        for hit in results:
        	print 'xxx';
        	print hit['content'].encode('utf-8');
