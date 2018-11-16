import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Method.LoadData import LoadData
from MkGraph.AccuracyPlot import AccuracyPlot

"""
This script use the KNN　algorithm to predict behavior label
reduced dimension algorithm is LDA
Loaded file: Data\Labeling\C\LNN_Train_data.xlsx
"""

# Variable
dim = [i+1 for i in range(5)]
all_accuracy = np.array([])

# Run the experiment from one dimension to five dimension
for element in dim:
    # Read file LNN_Train_data.xlsx to train/test
    org_data, org_label = LoadData.get_lnn_training_data()

    # Normalize the data
    normalized_data = preprocessing.normalize(org_data)
    # print(normalized_data)

    # Use LDA algorithm to reduce the dimensions
    lda = LinearDiscriminantAnalysis(n_components=element)
    lda.fit(normalized_data, org_label)
    reduced_data = lda.transform(normalized_data)

    normalized_data = preprocessing.normalize(reduced_data)
    print(normalized_data)

    X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)

    # Using KNN to classify
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)

    # Record accuracy
    accuracy = round(accuracy_score(y_test, predicted), 4)
    all_accuracy = np.append(all_accuracy, accuracy)

    # Record confusion matrix
    C_matrix = confusion_matrix(y_test, predicted)
    print('<---Dim is' + str(element), '--->')
    print('<---Confusion Matrix(KNN)--->\n', C_matrix)
    print('<---Accuracy: ',  accuracy * 100, '%--->')
    print('<----------------------------------------------->')

# Output the graph
path_name = '../Data/Graph/KNN_predicted.png'
path_name = os.path.join(os.path.dirname(__file__), path_name)
AccuracyPlot.build_accuracy_plot(
    'Accuracy vs KNN', np.array(dim), all_accuracy, path_name)




