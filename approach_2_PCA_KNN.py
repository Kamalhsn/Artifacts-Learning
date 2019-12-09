# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:47:00 2019

"""
import os
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from skimage import measure
#from skimage.measure import compare_ssim as ssim
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


dirName_under = r"D:\MRI\Oasis_data_new\Under";
dirName_fully = r"D:\MRI\Oasis_data_new\Fully";
stand_path_fully = r"D:\MRI\Oasis_data_new\Fully\OAS1_0060_MR1\OAS1_0060_MR1_mpr-1_anon.nii.gz" # path to fullysampled mri image
stand_path_under = r"D:\MRI\Oasis_data_new\Under\OAS1_0060_MR1\OAS1_0060_MR1_mpr-1_anon.nii.gz"

pca = decomposition.PCA(n_components=2)

def FileRead(file_path):
    nii = nib.load(file_path)
    data = nii.get_data()
    return data

def Nifti3Dto2D(Nifti3D):
    Nifti3DWOChannel = Nifti3D#[:,:,:,0] #Considering there is only one chnnel info
    Nifti2D = Nifti3DWOChannel.reshape(np.shape(Nifti3DWOChannel)[0], np.shape(Nifti3DWOChannel)[1] * np.shape(Nifti3DWOChannel)[2])
    return Nifti2D

def Nifti2Dto1D(Nifti2D):
    Nifti1D = Nifti2D.reshape(np.shape(Nifti2D)[0] * np.shape(Nifti2D)[1])
    return Nifti1D

def Nifti1Dto2D(Nifti1D, height):
    Nifti2D = Nifti1D.reshape(height,int(np.shape(Nifti1D)[0]/height))
    return Nifti2D

def Nifti2Dto3D(Nifti2D):
    Nifti3DWOChannel = Nifti2D.reshape(np.shape(Nifti2D)[0],np.shape(Nifti2D)[0],np.shape(Nifti2D)[1]//np.shape(Nifti2D)[0])
    return Nifti3DWOChannel

def FileSave(data, file_path):
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)
    
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def getListOfFiles(dirName): 
    listOfFile = os.listdir(dirName)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

scaler_under = StandardScaler()
scaler_under.fit(Nifti3Dto2D(FileRead(stand_path_fully)))
scaler_fully = StandardScaler()
scaler_fully.fit(Nifti3Dto2D(FileRead(stand_path_under)))

#pca_data= pca.fit(Nifti3Dto2D(FileRead(path_pca)))

def main_fully():
    listOfFiles_fully = getListOfFiles(dirName_fully)
    oned_fulsam = []
    listOfFiles_fully = []
    for (dirpath, dirnames, filenames) in os.walk(dirName_fully):
        listOfFiles_fully += [os.path.join(dirpath, file) for file in filenames]
    for elem_fully in listOfFiles_fully:
        red_files_fully = FileRead(elem_fully)
        red_files_fully.astype('float64')
        twod_fully = scaler_fully.transform(Nifti3Dto2D(red_files_fully))
        #ldd_fully = pca.fit_transform(twod_fully)
        #oned_fully= normalize(Nifti2Dto1D(twod_fully))
        oned_fulsam.append(twod_fully)
    oned_fulsam = np.asarray(oned_fulsam, dtype=np.float64, order='C')
    return oned_fulsam
oned_fulsam = main_fully()

def main_under():
    listOfFiles_under = getListOfFiles(dirName_under)
    oned_unsam = []
    twod_ud = []
    listOfFiles_under = []
    for (dirpath, dirnames, filenames) in os.walk(dirName_under):
        listOfFiles_under += [os.path.join(dirpath, file) for file in filenames]
    for elem_under in listOfFiles_under:
        red_files_under = FileRead(elem_under)
        red_files_under.astype('float64')
        twod_under = scaler_under.transform(Nifti3Dto2D(red_files_under))
        ldd_under = pca.fit_transform(twod_under)
        #oned_under= normalize(Nifti2Dto1D(twod_under))
        twod_ud.append(twod_under)
        oned_unsam.append(ldd_under)
    twod_ud = np.asarray(twod_ud, dtype=np.float64, order='C')
    oned_unsam = np.asarray(oned_unsam, dtype=np.float64, order='C')
    return oned_unsam, twod_ud
oned_unsam , twod_ud = main_under()

artifacts = twod_ud - oned_fulsam
artifacts = np.asarray(artifacts, dtype=np.float64, order='C')
art_data = []
i=0
for artifacts[i,:,:] in artifacts :
    #print(artifacts[i,:,:].shape)
    ldd_art = pca.fit_transform(artifacts[i,:,:])
    art_data.append(ldd_art)
art_data = np.asarray(art_data, dtype=np.float, order='C')
art_data = Nifti3Dto2D(art_data)
oned_unsam = Nifti3Dto2D(oned_unsam)

print("Shape of the artifacts",art_data.shape)
under_sampled_train,under_sampled_test, artifacts_train,artifacts_test = train_test_split(oned_unsam, art_data, test_size=0.2, random_state=0)
print('split completed')
"""#Finding K value
error_rate = []
for i in range(1,44):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(under_sampled_train,artifacts_train)
    pred_i = knn.predict(under_sampled_test)
    error_rate.append(np.mean(pred_i != artifacts_test))
#plt.figure(figsize=(10,6))
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import regression_report
plt.plot(range(1,44),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(under_sampled_train,artifacts_train)
pred = knn.predict(artifacts_test)
print('WITH K=1')
print('\n')
print(confusion_matrix(artifacts_test,pred))
print('\n')
######
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
rmse_val = [] #to store rmse values for different k
for K in range(44):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(under_sampled_train, artifacts_train)  #fit the model
    pred=model.predict(under_sampled_test) #make prediction on test set
    error = sqrt(mean_squared_error(artifacts_test,pred)) #calculate rmse
    R=rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
#plotting the rmse values against k value
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
rmse_val.sort()
print(rmse_val)
#print(regression_report(artifacts_test,pred))
"""
#K nearest neighbor
neigh = KNeighborsRegressor(n_neighbors=11)
neigh.fit(under_sampled_train , artifacts_train)
print('nearest neighbour fit completed')
data_predicted = neigh.predict(under_sampled_test)
print("Type of predicted data is : ",type(data_predicted))
#comaring ssim with predicted data
data_pred = data_predicted[1,:]
data_unsam_test = under_sampled_test[1,:]
data_fulsam_pred =  data_unsam_test - data_pred
y = Nifti1Dto2D(data_fulsam_pred , 256)
inv_pca_data = pca.inverse_transform(y)
threed_op = Nifti2Dto3D(inv_pca_data)

#data_fulsam_pred = np.diff([data_pred,data_unsam_test])
p= r"D:\MRI\Oasis_data_new\Fully\OAS1_0049_MR1\OAS1_0049_MR1_mpr-1_anon.nii.gz"
fully_45= FileRead(p)
fully_45 = np.asarray(fully_45, dtype=np.float64)
#from skimage.measure import compare_ssim as ssim
#import matplotlib.pyplot as plt
#import cv2
#'exec(%matplotlib inline)'
#def compare_image (threed_op, fully_45,title):
 #   s=ssim(threed_op, fully_45)
  #  fig=plt.figure("comparing")
    #plt.suptitle("SSIM: %.2f", %(s))
#threed_op
"""ax=fig.add_subplot(1,2,1)
plt.imshow(threed_op,cmap=plt.cm.gray)
plt.axis("off")
#fully_45
ax=fig.add_subplot(1,2,2)
plt.imshow(fully_45,cmap=plt.cm.gray)
plt.axis("off")
plt.show()
"""
print("ssim calculated = " ,measure.compare_ssim(threed_op, fully_45))

#Calculating root mean square error
def rmse(data_predicted, artifacts_test):
    differences = (data_predicted) - (artifacts_test)
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    print("RMSE is : ", rmse_val)
    return rmse_val
rmse(data_predicted, artifacts_test)
"""
#comaring ssim with predicted data
data_pred = (data_predicted[1,:], 256)
data_unsam_test = (under_sampled_test[1,:], 256)
data_fulsam_pred = data_pred - data_unsam_test
data_fulsam_pred_3d = Nifti2Dto3D(Nifti1Dto2D(data_fulsam_pred))
data_fulsam_pred_3d = normalize(data_fulsam_pred_3d)
data_fulsam_test = Nifti2Dto3D(Nifti1Dto2D(oned_fulsam[65,:], 256))
print("ssim calculated = " ,measure.compare_ssim(data_fulsam_pred_3d, data_fulsam_test))

#Calculating root mean square error
def rmse(data_predicted, artifacts_test):
    differences = (data_predicted) - (artifacts_test)
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    print("RMSE is : ", rmse_val)
    return rmse_val
rmse(data_predicted, artifacts_test)
"""
