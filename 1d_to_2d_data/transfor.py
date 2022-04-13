import os
import numpy as np

data_path="D:\\LenovoQMDownload\\PythonProject\\SVM_SMO\\data_Split"
images_path=os.path.join(data_path, "Images")#txt数据的路径
labels_path=os.path.join(data_path,"Labels")#标签路径
#print(images_path)#输出Images路径
for item in os.listdir(images_path):
    item_path=os.path.join(images_path,item)#将数据路径和数据名连起来
    item_label_path=os.path.join(labels_path,item)#将标签路径和标签名连起来
    #print(item_label_path)
    #print(item_path)
    data_1D=np.loadtxt(item_path,dtype=int,delimiter=',')#一个txt一个[]
    #print(data_1D.tolist())
    data_label_1D=np.loadtxt(item_label_path,dtype=int)
    #print(data_label_1D)
    var=np.concatenate([data_1D,data_label_1D],axis=0)
    print(*var)
    f = open('D:\\LenovoQMDownload\\PythonProject\\SVM_SMO\\data_Split\\2d_400_data.txt', 'a')
    f.write(', '.join(map(str, var)) + '\n')
    f.close()












