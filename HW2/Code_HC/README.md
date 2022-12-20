# README

## Homework2 (超啟發式演算法（Meta-heuristic Algorithm ))

### (1)	爬山演算法( Hill climbing, HC )

1. ***Import 所需之相關模組:***

    - `import numpy as np` : 用於操作陣列

---

2. ***建立所需之相關參數:***

    - `MAX_ITERATION = 500` : 迭代次數

---

3. ***載入資料( Capacity, Profit, Weight ):***

    ```py
    # Load the data
    path = '../Data Set/'
    data_lst = ['p07_c.txt', 'p07_p.txt', 'p07_w.txt']

    with open(path + data_lst[0], 'r') as f:
        CAPACITY = int(f.read())
        print('Capacity:', CAPACITY)

    PROFIT_LST = np.array([])
    with open(path + data_lst[1], 'r') as f:
        for line in f:
            PROFIT_LST = np.append(PROFIT_LST, int(line))

    WEIGHT_LST = np.array([])
    with open(path + data_lst[2], 'r') as f:
        for line in f:
            WEIGHT_LST = np.append(WEIGHT_LST, int(line))

    # Initialize the selection
    selection_lst = np.array([1 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0, 0])

    print('    |Profit    ', '|Weight    ', '|Selection')
    print('--------------------------------------')
    for i in range(len(selection_lst)):
        print('{:<3} |{:<9}  |{:<9}  |{:<9} '.format(str(i) + '.', PROFIT_LST[i], WEIGHT_LST[i], selection_lst[i]))
    ```

    ![profit_list_HC](https://github.com/tsungru/NCHU-HW/blob/main/HW2/Figure/profit_list_HC.png?raw=true)

    - 將 Capacity、 Profit、 Weight 分別存入 `CAPACITY`、 `PROFIT_LST`、 `WIEGHT_LST`。

    - 初始化 Selection 為 `selection_lst`。

    - 並將其對應之 Profit, Weight, 與是否被 `selection_lst` 選擇印出。

---

4. ***用於根據 Selection 計算對應的 Profit***

    ```py
    # The function of calculating the profit by corresponding selection
    def count_profit(selection_lst):
        total_profit = 0
        for i in range(len(selection_lst)):
            if selection_lst[i]:
                total_profit = total_profit + PROFIT_LST[i]
        
        return total_profit
    ```

---    

5. ***用於根據 Selection 計算對應的 Weight***

    ```py
    # The function of calculating the weight by corresponding selection
    def count_weight(selection_lst):
        total_weight = 0
        for i in range(len(selection_lst)):
            if selection_lst[i]:
                total_weight = total_weight + WEIGHT_LST[i]
        
        return total_weight  
    ```

---    

6. ***用於更改 Selection***

    ```py
    # The function of changing the selection
    def change_selection(selection_lst, idx):
        if selection_lst[idx] == 0:
            selection_lst[idx] = 1
        else:
            selection_lst[idx] = 0
    ```

---    

7. ***尋找最佳解***

    ```py
    # Find the optimal solution
    SIZE = len(selection_lst)

    # The weight and profit of initial selection
    total_weight = count_weight(selection_lst)
    total_profit = count_profit(selection_lst)

    # Used to record profits
    profit_record = np.array([total_profit])

    for i in range(MAX_ITERATION):
        sel = np.random.randint(0, SIZE)
        change_selection(selection_lst, sel)
        temp_weight = count_weight(selection_lst)
        temp_profit = count_profit(selection_lst)
        
        # If the new selection can satisfy the constraint of weight and can have a better profit 
        if (temp_weight <= CAPACITY and temp_profit >= total_profit):
            total_weight = temp_weight
            total_profit = temp_profit
        else:
            change_selection(selection_lst, sel)
        
        # Record the profit
        profit_record = np.append(profit_record, total_profit)

    print("Weight:", total_weight)
    print("Profit:", profit_record[-1])
    ```

    ![final_value_HC](https://github.com/tsungru/NCHU-HW/blob/main/HW2/Figure/final_value_HC.png?raw=true)

    - 將初始 Selection 的 Weight 與 Profit 儲存至 `total_weight` 與 `total_profit`。
    
    - 進行迭代時，隨機選取任意一個物品的代號 `sel`。

    - 計算新 Selection 的 Weight 與 Profit，`temp_weight` 與 `temp_profit`。

    - 如果新 Selection 可以滿足 Capacity 的限制並且有較好的 Profit，即可進行選擇。反之，繼續尋找新的 Selection。

    - 紀錄每一次迭代 Profit 的更新至 `profit_record`。

--- 

8. ***輸出結果***

    ```py
    # Output the profit growth rate
    import matplotlib.pyplot as plt

    plt.figure() 
    plt.plot(profit_record, 'r') 
    plt.title('Profit growth rate') 
    plt.xlabel('Iteration number') 
    plt.ylabel('Profit') 
    plt.show()
    ```

    ![output_HC](https://github.com/tsungru/NCHU-HW/blob/main/HW2/Figure/output_HC.png?raw=true)

