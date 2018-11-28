# Method 1
# Step1, use the Preprocess/SplitExcel.py to remove the unreasonable data and get partial excel data
# Step2, use the Preprocess/Build264Data.py to generate the 264 dimension data
# Step3, use the Preprocess/BulidTrainData.py to generate the FNN　training data
# Step4, use Main.py to predict the label by fnn, lnn, DNN

# Method 2
# <---Important information--->
# If the the performance of the neural networks is poor
# We need to split the Original_data.xlsx with different label(C1~C6) by Preprocess/SplitOriginalData.py
# And, we start to seperate the data set by Preprocess/ObserveDataSet.py
# To improve the performance
# Step1, use the Preprocess/SplitExcel.py to remove the unreasonable data and get partial excel data
# Step2, use the Preprocess/Build264Data.py to generate the 264 dimension data
# step3, use the Preprocess/SplitExcel.py to separate the label of the C1 ~ C6
# Step4, use the Preprocess/RefactorOriginalData.py to refactor
# The file from C1_Original_data.xlsx to C6_Original_data.xlsx
# Step5, use the Preprocess/BulidTrainData.py to generate the FNN　training data
# Step6, use Main4.py to predict the label by fnn


