import numpy as np

txt_path_2d_no_label=open('D:/LenovoQMDownload/PythonProject/SVM_SMO/1d_to_2d_data/2d_no_label.txt')
charact_data_mean = []
charact_data_var=[]

for line in txt_path_2d_no_label.readlines():
    lineArr = line.strip().split(',')
    charact_data_mean.append(np.mean([float(data) for data in lineArr[:]]))
    charact_data_var.append(np.var([float(data) for data in lineArr[:]]))

    #data_Set.append()
    #print(np.mean(lineArr,1))

print(charact_data_mean)
print(charact_data_var)
















