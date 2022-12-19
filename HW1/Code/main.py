import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

base_dir = 'C:/Users/user/Desktop/homework/HW1/Data Set/flowers/'
cat_lst = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Hyper parameter
imgsize = 224
X = []
y = []

def load_data(data_dir, cat):
    data_dir = data_dir + cat + '/'
    print ("Loading：", data_dir) 
    print()
    for img in os.listdir(data_dir): 
        path = data_dir + img
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            print('Wrong path')
            exit(0)
        img = cv2.resize(img, (imgsize, imgsize)) 
        X.append(np.array(img)) 
        y.append(cat) 

def show_history(history): # 显示训练过程中的学习曲线
    loss = history.history['loss'] #训练损失
    val_loss = history.history['val_loss'] #验证损失
    epochs = range(1, len(loss) + 1) #训练轮次
    plt.figure(figsize=(12,4)) # 图片大小
    plt.subplot(1, 2, 1) #子图1
    plt.plot(epochs, loss, 'bo', label='Training loss') #训练损失
    plt.plot(epochs, val_loss, 'b', label='Validation loss') #验证损失
    plt.title('Training and validation loss') #图题
    plt.xlabel('Epochs') #X轴文字
    plt.ylabel('Loss') #Y轴文字
    plt.legend() #图例
    acc = history.history['acc'] #训练准确率
    val_acc = history.history['val_acc'] #验证准确率
    plt.subplot(1, 2, 2) #子图2
    plt.plot(epochs, acc, 'bo', label='Training acc') #训练准确率
    plt.plot(epochs, val_acc, 'b', label='Validation acc') #验证准确率
    plt.title('Training and validation accuracy') #图题
    plt.xlabel('Epochs') #X轴文字
    plt.ylabel('Accuracy') #Y轴文字
    plt.legend() #图例
    plt.show() #绘图

if __name__ == "__main__":

    # Load Data 
    for i in range(len(cat_lst)):
        load_data(base_dir, cat_lst[i])
    
    # Label and normalize the Data
    X = np.array(X)
    X = X / 255
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y, 5)

    # Divide the data into 'training set' and 'testing set'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#############################################################################     

    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (imgsize, imgsize, 1)))                    
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 
    cnn_model.add(layers.MaxPooling2D((2, 2))) 
    cnn_model.add(layers.Flatten()) 
    cnn_model.add(layers.Dense(512, activation = 'relu')) 
    cnn_model.add(layers.Dropout(0.2)) 
    cnn_model.add(layers.Dense(5, activation='softmax')) 
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['acc']) 

#############################################################################

    # cnn_model = models.Sequential()
    # cnn_model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (imgsize, imgsize, 1)))                    
    # cnn_model.add(layers.MaxPooling2D((2, 2)))
    # cnn_model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 
    # cnn_model.add(layers.MaxPooling2D((2, 2))) 
    # cnn_model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    # cnn_model.add(layers.MaxPooling2D((2, 2))) 
    # cnn_model.add(layers.Flatten()) 
    # cnn_model.add(layers.Dense(512, activation = 'relu')) 
    # cnn_model.add(layers.Dropout(0.2)) 
    # cnn_model.add(layers.Dense(5, activation='softmax')) 
    # cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['acc']) 

#############################################################################

    # 训练网络并把训练过程信息存入history对象
    history = cnn_model.fit(X_train, y_train, epochs=40, validation_split=0.2)

    show_history(history) # 调用这个函数

    result = cnn_model.evaluate(X_test, y_test) #评估测试集上的准确率
    print('CNN的测试准确率为',"{0:.2f}%".format(result[1]))    

