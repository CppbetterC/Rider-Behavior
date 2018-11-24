"""
這是個對於在安全帽上面收集到的資料進行取特徵值的動作的程式碼
這個 python script 把 Data/Labeling/C/Split_data.xlsx
用 Formula.py 的公式去轉換成坐標點的資料集
轉換成有264維度的原始資料集(88+88+88)
是每 2、4和6 筆取出的資料用公式計算後
取出的特徵值 (Feature)
窗格大小在移動的時候必須重疊與前一筆資料重疊(1/2)的
並命名為 Original_data.xlsx
"""

import os
import numpy as np
import pandas as pd

from Method.LoadData import LoadData
from Method.DataCombine import DataCombine
from Method.Export import Export

label_type = 'C'

load = LoadData()
load_data, org_label = load.get_original_excel(label_type, 'Split_data')


# Convert the original data to the data with 264 dimensions
org_data = DataCombine.combine(load_data)
print(org_data.shape)
print(org_label.shape)

header = ['Dim'+str(i) for i in range(1, 265, 1)]
pd_data = pd.DataFrame(org_data, columns=header)
pd_label = pd.DataFrame(org_label, columns=['Label'])

result = pd.concat([pd_data, pd_label], axis=1)

print(result.head())

path_name = '../Data/Labeling/' + label_type + '/' + 'Original_data.xlsx'
path_name = os.path.join(os.path.dirname(__file__), path_name)
Export.export_org_data(result, path_name)
print('<---Generate the 264 Dimension Data--->')
