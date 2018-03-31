import preprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt

#==============================================================================
test_f = open("test_images.txt", "r")
test_data = test_f.readlines()
test_f.close()

test_f2 = open("test_labels.txt", "r")
test_labels = test_f2.readlines()
test_f2.close()

train_f = open("train_images.txt", "r")
train_data = train_f.readlines()
train_f.close()

train_f2 = open("train_labels.txt", "r")
train_labels = train_f2.readlines()
train_f2.close()

test_data = test_data[0].split()
test_labels = test_labels[0].split()
train_data = train_data[0].split()
train_labels = train_labels[0].split()

temp_img = cv2.imread("/media/thesis/work/tryonline/ML_Lab/Bangla Handwritten Number OCR/all_data/all_data/a.jpg")
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
temp_img = cv2.resize(temp_img, (28, 28))

print("Jinishpotro loaded successfully !!")
#======================================================================


data = train_data
labels = train_labels

def getArr(img):
      
    height = np.size(img, 0)
    width = np.size(img, 1) 
    
    arr = []
    
    i=0
    
    while i<height:
    
        j=0
        
        while j<width:
            
            px = img[i, j]
            if(px>0):
                px=255
            
            px/=255
            
            arr.append(px)
            
            j = j+1
        i = i+1
    
    return arr

def showImage(img):
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.show()
    
    return

def getImg(arr):
    
    cnt=0
    for i in range(0, 28):
        for j in range(0, 28):
            temp_img[i, j] = arr[cnt]
            cnt+=1
    
    return temp_img


def getPreprocessed(arr2):
    
    img = getImg(arr2)
    img = 255-img
    img = preprocess.func(img)
    img = 255-img
    img = cv2.resize(img, (28, 28))
    
    return getArr(img)
    







arr = []
temp = []
cnt = 0

for i in range(0, len(data)):

    temp.append(int(data[i]))
    
    cnt+=1
    if cnt==784:
        arr.append(temp)
        temp = []
        cnt=0


        
f = open('normal2.csv','w')
#
for i in range(0, len(arr)):
    
    arr2 = arr[i] 
    name = labels[i]
    
#    arr2 = getPreprocessed(arr2)
    
    print(name)
    for j in range(0, len(arr2)):

    	now = arr2[j]/255
        if(now!=0):
        	now=1.0
       	f.write(str(now)+', ')

       

    f.write(str(name)+'\n')
f.close()

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)

# clf.fit(train_data,train_labels);
# clf.predict(test_data,test_labels);


