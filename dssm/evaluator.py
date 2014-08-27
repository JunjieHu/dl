#encoding=utf-8
import sys,math
from collections import OrderedDict

class Evaluator():
    def __init__(self, th = lambda x: int(x>1)):
        self.th = th

    def dcg(self, li, pos):
        gain = 0.0
        dcg = []
        for i, label in enumerate(li, start=1):
            gain += (2 ** label - 1)/math.log(i+1, 2)
            dcg.append(gain)

        ans = []
        for p in pos:
            if p > 0 and p <= len(dcg):
                ans.append(dcg[p-1])
            else:
                ans.append(gain)
        return ans

    def idcg(self, li, pos):
        return self.dcg(sorted(li, reverse=True), pos)

    def ndcg(self, li, pos):
        dcg = self.dcg(li, pos)
        idcg = self.idcg(li, pos)
        ndcg = []
        for d, i in zip(dcg, idcg):
            if i > 0:
                ndcg.append(d/i)
            else:
                ndcg.append(0)
        return ndcg

    def indcg(self, li, pos):
        id = self.idcg(li, pos)
        for p in id:
            if id[p] > 0:
                id[p] = 1.0
            else:
                id[p] = 0.0
        return id

    def precision(self, li, pos):
        k = 0.0
        p = {}
        for i, label in enumerate(li,start=1):
            k += self.th(label)
            if i in pos:
                p[i] = k/i
        return p

    def iprecision(self, li, pos):
        return self.precision(sorted(li, reverse=True), pos)

    def ap(self, li, pos):
        a = 0.0
        k = 0.0
        for i, label in enumerate(li, start=1):
            k += self.th(label)
            a += self.th(label)*k/i
        if k == 0:
            res = 0
        else:
            res = a/k
        ap = {}
        for p in pos:
            ap[p] = res
        return ap

    def iap(self, li, pos):
        return self.ap(sorted(li, reverse=True), pos)

def load_results(file):
	labels = [];
	qids = [];
	scores = [];
	for line in open(file, 'r'):
		li = line.split(' ');
		qids.append(li[0]);
		scores.append(float(li[1]));
		labels.append(int(li[2]));

	return labels, qids, scores;
def main(total):
    labels, qids, scores = load_results(sys.argv[1]);
    assert len(qids) == len(labels)
    eval = Evaluator()
    measures = ['dcg','ndcg']
    pos = [1,3,5,10]

    data = OrderedDict()
    for i in xrange(len(qids)):
        qid = qids[i]
        if qid in data:
            data[qid][0].append(scores[i])
            data[qid][1].append(labels[i])
        else:
            data[qid] = {}
            data[qid] = ([scores[i]], [labels[i]])

    eval_res = OrderedDict()
    for me in measures:
        eval_res[me] = OrderedDict()
        for p in pos:
            eval_res[me][p] = []

    for score, label in data.values():
        sorted_label, sorted_score = zip(*sorted(zip(label, score), key=lambda x:x[1], reverse=True))
        #print sorted_score, sorted_label
        for me in measures:
            res = getattr(eval, me)(sorted_label, pos)
            for i, p in enumerate(pos):
                eval_res[me][p].append(res[i])
            #print '\t'.join(['%s@%d: %.3f'%(me, p, res[i]) for i, p in enumerate(pos)])
    #print eval_res['dcg'][1]
    for me, v  in eval_res.items():
        print '\t'.join(['%s@%d'%(me, p) for p in v.keys()])
        print '\t'.join(['%.3f'%(sum(li)/len(li)) for p, li in v.items()])
        print '\t'.join(['%.3f'%(sum(li)/total) for p, li in v.items()])


if __name__ == "__main__":
    total = int(sys.argv[2]);
    main(total);
    '''
    eval = Evaluator()
    measures = ['dcg','ndcg','precision','ap','idcg','indcg','iprecision','iap']
    measures = ['precision']
    li = [3,2,3,0,1,2]
    print li
    pos = [1,3,5]
    for me in measures:
        s = getattr(eval, me)(li, pos)
        for p in pos:
            print "%s@%d:%.3f" % (me, p, s[p])
    '''
