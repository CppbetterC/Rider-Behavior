# Clear the data
# Step1, use the Preprocess/SplitExcel.py to remove the unreasonable data and get partial excel data
# Step2, use the Preprocess/Build264Data.py to generate the 264 dimension data
# Step3, use the Preprocess/BulidTrainData.py to generate the FNN　training data
# Step4, use Main.py to predict the label by fnn, lnn, DNN

# <---Important information--->
# If the the performance of the neural networks is poor
# We need to split the Original_data.xlsx with different label(C1~C6) by Preprocess/SplitOriginalData.py
# And, we start to seperate the data set by Preprocess/ObserveDataSet.py
# To improve the performance


