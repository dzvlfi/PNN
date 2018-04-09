
# coding: utf-8
#Dzulfiqar Ridha 1301154298 IF-39-04
# In[75]:


import matplotlib.pyplot as plt
import math
import numpy as np
import operator

#load data
data_train = np.genfromtxt("data_train_PNN.txt",skip_header=1)
data_test = np.genfromtxt("data_test_PNN.txt",skip_header=1)

#visualization data train
xyz = plt.figure().add_subplot(111, projection='3d')

xyz.set_xlabel("att1")
xyz.set_ylabel("att2")
xyz.set_zlabel("att3")

xyz.scatter(data_train[:,0], data_train[:,1], data_train[:,2], c=data_train[:,3])

plt.show()


# In[76]:


#menghitung PDF
def PDF(test,train,s):
    return math.exp(-((((test[0]-train[0])**2)+((test[1]-train[1])**2)+((test[2]-train[2])**2))/(2*(s)**2)))


# In[77]:


#Klasifikasi
def klasifikasi(dtest,dtrain,s):
    kelas = {0:0.0,1:0.0,2:0.0}
    hasil = []
    for test in dtest:
        for train in dtrain:
            kelas[int(train[3])] = kelas[int(train[3])] + PDF(test,train,s)

        hasil.append(max(kelas.iteritems(), key=operator.itemgetter(1))[0])
        kelas = {0:0.0,1:0.0,2:0.0}
    return np.array(hasil)


# In[78]:


#menghitung akurasi
def tepat():
    n=0
    smooth = []
    persenan = []
    while n<1:
        n+=0.05
        akurasi = klasifikasi(data_train,data_train,n)
        sum = 0
        for i in range(len(data_train)):
            cek = akurasi[i] == int(data_train[:,3][i])
            if cek:
                sum+=1
        persen = float(sum)/len(data_train)*100
        persenan.append(persen)
        smooth.append(n)
        print "smoothing:",n,", akurasi:",persen
    print "nilai smoothing terbaik:",smooth[persenan.index(max(persenan))],"dengan akurasi: ",max(persenan)
    plt.xlabel("nilai smoothing")
    plt.ylabel("akurasi (%)")
    plt.plot(smooth,persenan)
    plt.show()
    return smooth[persenan.index(max(persenan))]


# In[79]:


if __name__ == '__main__':
    kelas = np.concatenate((data_test,klasifikasi(data_test,data_train,tepat())[:,None]),axis=1)
    
    #visualisasi hasil======
    abc = plt.figure().add_subplot(111, projection='3d')

    abc.set_xlabel("att1")
    abc.set_ylabel("att2")
    abc.set_zlabel("att3")

    abc.scatter(kelas[:,0], kelas[:,1], kelas[:,2], c=kelas[:,3])
    plt.show()
    #visualisasi hasil======
    
    f = open('prediksi.txt','w')
    f.write("\n".join(map(lambda x: str(x), kelas)))
    f.close()

