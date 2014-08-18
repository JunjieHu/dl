#!/usr/local/bin/python
#-*- encoding:utf-8 -*-

from whoosh.index import create_in
from whoosh.fields import *
from chinesetokenizer import ChineseAnalyzer
import ConfigParser

analyzer = ChineseAnalyzer();
config = ConfigParser.ConfigParser();
config.read('config');

f = formats.Frequency();
schema = Schema(appid = ID(stored=True), content=TEXT(stored=True, analyzer=analyzer,vector=f), description=TEXT(stored=True, analyzer=analyzer, vector=f));
ix = create_in(config.get('bm25', 'MulFieldsIndex'), schema)

writer = ix.writer()
f = open(config.get('bm25', 'corpus'));
for line in f:
    li = line.split('\t');
    if len(li) != 4:
        continue;
    if li[2].strip() == '':
        description = u' ';
    else:
        description = li[2].decode('utf-8');
    writer.add_document(appid=li[0].decode('utf-8'),content=li[3].decode('utf-8'), description=description);
writer.commit()

f.close();