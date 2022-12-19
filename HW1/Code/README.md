# README

## Homework1 (深度學習（Deep Learning）)

1. ***Import 所需之相關模組:***

    - `from tensorflow.keras import layers, models` : 用於建立神經網路

    - `import matplotlib.pyplot as plt` : 用於視覺化訓練節果

    - `import os` : 用於執行系統指令

    - `import cv2` : 用於處理影像

    - `import numpy as np` : 用於操作陣列

---

2. ***建立所需之相關參數***

    - `base_dir = 'C:/Users/Owner/OneDrive//Desktop/homework/HW1/Data Set/flowers/'` : 檔案位置

    - `cat_lst = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']` : 所有 Label 種類

    - `X = []` : Data 之集合

    - `y = []` : Label 之集合

    - `imgsize = 150` : 圖片所需之大小

    - `batch_size = 128` : 樣本數大小

    - `epochs = 50` : 迭帶訓練次數

---

3. ***讀取檔案***

    ```py
    def load_data(data_dir, cat):
        data_dir = data_dir + cat + '/'
        print ("Loading：", data_dir) 
        print()
        for img in os.listdir(data_dir): 
            path = data_dir + img
            img = cv2.imread(path) 
            if img is None:
                print('Wrong path')
                exit(0)

            # Resize all data
            img = cv2.resize(img, (imgsize, imgsize)) 
            X.append(np.array(img)) 
            y.append(cat) 
    ```
    
    - 利用 Function `load_data` 並傳入資料集位置 `data_dir` 與對應之 Label `cat` ，讀取圖像資料並儲存至 `X` 與 `y` 中。

---

4. ***進行資料前處理***

    ```py
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical

    # Load all data and the labels into the array X and y, respectively 
    for i in range(len(cat_lst)):
        load_data(base_dir, cat_lst[i])

    # Normalize the X 
    X = np.array(X)
    X = X / 255

    # Transform the y from the format of string to the format of array with binary elements 
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y) 
    y = to_categorical(y, 5)
    ```

    - 為使得模型更加易於訓練，可將 Data 進行正規化處理。

    - 將 `X` 中的所有 Data 進行正規化，使其中的每一個值介於 [ 0, 1 ]。

    - 將 `y` 中的所有 Label 的標籤由字串轉為由 One hot encoding 表示。

---

5. ***切割資料集***

    ```py
    from sklearn.model_selection import train_test_split

    # Divide the data into 'training set' and 'testing set'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    ```

    - 在進行訓練前，需將資料集進行切割。這裡將資料集中的 80% 切分為用於訓練模型的 `X_train` 與 `y_train`，以及 20% 切分為用於測試模型的 `X_test` 與 `y_test`。

---

6. ***CNN 模型***

    ```py
    # Build the model by CNN
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu', input_shape = (imgsize, imgsize, 3)))                    
    cnn_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    cnn_model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu')) 
    cnn_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))) 

    cnn_model.add(layers.Conv2D(filters = 96, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
    cnn_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))) 

    cnn_model.add(layers.Conv2D(filters = 96, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
    cnn_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))) 

    cnn_model.add(layers.Flatten()) 
    cnn_model.add(layers.Dense(512, activation = 'relu')) 
    cnn_model.add(layers.Dropout(0.5)) 
    cnn_model.add(layers.Dense(5, activation='softmax')) 
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['acc']) 
    ```

    - 所使用的神經網路架構。

---

7. ***開始進行訓練***

    ```py
    # Train the model by the training set
    history = cnn_model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
    ```

    - 利用 `X_train` 與 `y_train` 進行訓練，並調整 `batch_size` 與 `epochs`。

---

8. ***輸出訓練結果***

    ```py
    # The function of showing the training history of 'loss' and 'accruacy'
    def show_history(history): 

        # The image of training and validation loss
        loss = history.history['loss'] 
        val_loss = history.history['val_loss'] 
        epochs = range(1, len(loss) + 1) 
        plt.figure(figsize=(12,4)) 
        plt.subplot(1, 2, 1) 
        plt.plot(epochs, loss, 'r', label='Training loss') 
        plt.plot(epochs, val_loss, 'b', label='Validation loss') 
        plt.title('Training and validation loss') 
        plt.xlabel('Epochs') 
        plt.ylabel('Loss') 
        plt.legend() 

        # The image of training and validation accruacy
        acc = history.history['acc'] 
        val_acc = history.history['val_acc'] 
        plt.subplot(1, 2, 2) 
        plt.plot(epochs, acc, 'r', label='Training acc') 
        plt.plot(epochs, val_acc, 'b', label='Validation acc') 
        plt.title('Training and validation accuracy') 
        plt.xlabel('Epochs') 
        plt.ylabel('Accuracy') 
        plt.legend() 
        plt.show() 
    ```

    - 利用 `matplotlib` 將訓練結果 `history` 輸出。
    - 由兩個子圖分別表示 Training 與 Validation 的 Loss 與 Accuracy。

---

9. ***訓練結果***

    ```py
    # Show the image of training history 
    show_history(history) 
    ```

    ![Training Result]()

    - 如上圖所示，雖然在測試集上有良好的表現，但在驗證集上的效果並不佳，並且驗證集的 Loss 持續上升。由此可得出，此模型產生了 Overfitting 的現象。

---

10. ***模型評估***

    ```py
    # Evulate the CNN model by four different metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    y_true = []
    y_pred = []
    y_test_pred = cnn_model.predict(X_test)

    # Transform the real label and predicted label from the format of array to the format of index
    for i in range(len(y_test)):
        y_true.append(y_test[i].tolist().index(1.0))
        y_pred.append(y_test_pred[i].tolist().index(max(y_test_pred[i])))

    # The metric of accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # The metric of precision
    precision = precision_score(y_true, y_pred, average = 'macro')

    # The metric of recall
    recall = recall_score(y_true, y_pred, average = 'macro')

    # The metric of F1
    f1 = f1_score(y_true, y_pred, average = 'macro')

    print('The \"Accruacy\" of CNN model is {0:.5f}'.format(accuracy))    
    print('The \"Precision\" of CNN model is {0:.5f}'.format(precision))  
    print('The \"Recall\" of CNN model is {0:.5f}'.format(recall)) 
    print('The \"F1\" of CNN model is {0:.5f}'.format(f1)) 
    ```

    ![Classification Metrics](https://github.com/tsungru/NCHU-HW/blob/main/HW1/Figure/metric.png?raw=true)

---

11. ***將測試集中隨機十筆預測後的結果和正解進行比較***

    ```py
    # Randomly select ten different indexes
    idxs = np.random.randint(0, len(y_test), size = 10)
    real = []
    predict = []

    for i in range(10): 
        real.append(y_test[idxs[i]].tolist()) 
        predict.append(y_test_pred[idxs[i]].tolist())

    # Output real answers and predicted answers of ten different testing data
    for i in range(10):
        if i == 0:
            print("-" * 63)               
            print("| {:<20}| {:<20}| {:<20}".format("Real Answer", "Predicted Answer", "Predicted Success"))
            print("-" * 63)
        r_ans = cat_lst[real[i].index(max(real[i]))]
        p_ans = cat_lst[predict[i].index(max(predict[i]))]
        if r_ans == p_ans: 
            ans = "true"
        else:
            ans = "false"
        print("| {:<20}| {:<20}| {:<20}".format(r_ans, p_ans, ans))
    ```

    ![Compare Result](https://github.com/tsungru/NCHU-HW/blob/main/HW1/Figure/compare.png?raw=true)    

---

12. ***想法與結論***

    本次的作品是本人第一次接觸 CNN，在設計以及使用上稍有生疏，雖然能夠將模型訓練成功，但仍然無法解決嚴重 Overfitting 的問題。參考網路上與朋友的意見，將神經網路的層數增加、將Epoch數調高、將圖片改為灰階 ... 等等，皆無法有效改善 Overfitting 的現象，或是使的準確率低下，仍需精進自己並加以改良模型之設計，才可達到更好的表現。