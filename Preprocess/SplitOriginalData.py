import pandas as pd

from Method.LoadData import LoadData


header = ['Dim' + str(i) for i in range(1, 265, 1)]
column = header.append('Label')
org_data = LoadData.get_org_data()
pd_data = pd.DataFrame(org_data, columns=column)

for i in range(1, 7, 1):
    result = pd_data.loc[pd_data['Label'] == 'C'+str(i), header]
    # pd_array = pd.DataFrame(tmp_data, columns=column)
    result.to_excel('../Data/Labeling/C/Split_C'+str(i)+'Original_data.xlsx', sheet_name='Data', index=False)
    print('<---C' + str(i) + ' Successfully--->')

# print(pd_data)
