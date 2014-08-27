import cPickle as pickle
ofile = open('./2000000.0paras.pkl','rb')
params=pickle.load(ofile)
ofile.close()

c_w0, c_b0, c_w1, c_b1, c_w2, c_b2, c_w3, c_b3, tmp1,tmp2,tmp3,tmp4 = params

clean = [c_w0, c_b0, c_w1, c_b1, c_w2, c_b2, c_w3, c_b3]
ofile=open("2Cparas.pkl",'wb')
pickle.dump(clean, ofile)
ofile.close()
