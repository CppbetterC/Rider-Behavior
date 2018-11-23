import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, BatchNormalization, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from Method.LoadData import LoadData


# Load the file
org_data, org_label = LoadData.get_split_data()
# print('org_data', org_data.shape)
# print('org_label', org_label)

# random_state, 確保每次切分資料的結果都相同
X_train, X_test, y_train, y_test = \
    train_test_split(org_data, org_label, test_size=0.3, random_state=42)

# X_train = X_train.reshape((X_train.shape[0],60))
# X_test = X_test.reshape((X_train.shape[0],60))

# Building Model
# model1
print('<---Model START--->')
batch_size = 32
model = Sequential()

model.add(LSTM(128, dropout=0.5, input_dim=len(X_train[0]),
         activation='sigmoid', return_sequences=False))
model.add(Dense(32, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
adam = Adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
##
history = model.fit(X_train, y_train, epochs=500, batch_size=batch_size,
                    validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

result = model.predict(X_test[:10])
answer = y_train[:10]
ans = y_test[:10]
for x, y in zip(result, answer):
    print(x, ' ', y)

print('<---Model END--->')

# Predict,


# # model 3
# batch_size = 32
# model = Sequential()
# model.add(LSTM(128, dropout=0.2, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
# model.add(LSTM(32, return_sequences=True))
# model.add(Flatten())
# model.add(Dense(32))
# # model.add(BatchNormalization())
# model.add(Activation('sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# adam = Adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.summary()
#
# history = model.fit(X_train, y_train, epochs=500, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2,
#                     shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
# # In[ ]:
#
#
# # model 2
# batch_size = 32
# model = Sequential()
# model.add(LSTM(128, dropout=0.2, input_shape=(X_train.shape[1], X_train.shape[2]), activation='sigmoid',
#                return_sequences=True))
# model.add(LSTM(32, dropout=0.2, return_sequences=True, activation='sigmoid'))
# model.add(LSTM(32, dropout=0.2, return_sequences=True, activation='sigmoid'))
# model.add(LSTM(1, activation='softmax'))
# adam = Adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.summary()
#
# history = model.fit(X_train, y_train, epochs=500, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2,
#                     shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
# # In[ ]:
#
#
# # LSTM_Binary
# # model = Sequential()
# # model.add(LSTM(56,
# #                batch_input_shape=(None, 20, 3),
# #                unroll=True))
#
# # model.add(Dense(1))
# # model.add(Activation('softmax'))
#
# # adam = Adam(lr=0.001)
# # model.compile(optimizer=adam,
# #               loss='sparse_categorical_crossentropy',
# #               metrics=['accuracy'])
# # model.summary()
#
#
# # In[ ]:
#
#
# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
#
# # In[ ]:
#
#
# result = model.predict(X_test[:10])
# ans = y_test[:10]
# for i in range(len(result)):
#     print("%.f ,         %f" % (ans[i], result[i][0]))
# result
#
# # In[ ]:
#
#
# pFall = 0
# for index, i in enumerate(result):
#     print('data: {}'.format(ans[index]))
#     pFall = i[0] * 100
#     print('Fall：{}%'.format(pFall))
#     print('Nothing：{}%'.format(100 - pFall))
#
# # In[ ]:
#
#
# result = model.predict(X_train[:10])
# y_train[:10], result
#
# # In[ ]:
#
#
# pFall = 0
# for index, i in enumerate(result):
#     print('data: {}'.format(index))
#     pFall = i[0] * 100
#     print('Fall：{}%'.format(pFall))
#     print('Nothing：{}%'.format(100 - pFall))
#
# # In[ ]:
#
#
# from keras.models import load_model
#
# model = load_model('save_model.h5')
#
# # In[ ]:
#
#
# result = model.predict(X_test[:10])
# ans = y_test[:10]
# for i in range(len(result)):
#     print("%.f ,         %f" % (ans[i], result[i][0]))
# result
#
# # In[ ]:
#
#
# result = model.predict(X_train[:10])
# ans = y_train[:10]
# for i in range(len(result)):
#     print("%.f ,         %f" % (ans[i], result[i][0]))
# result
#
# # In[ ]:
#
#
# t = "[0.05496032536029816, -0.0059708356857299805, 0.005880177021026611]", "[0.018425151705741882, 0.018444299697875977, -0.08608400821685791]", "[0.02047291398048401, 0.016492068767547607, -0.08994460105895996]", "[0.027894631028175354, 0.026775777339935303, -0.009342879056930542]", "[0.030027076601982117, 0.01520085334777832, -0.03099125623703003]", "[1.3687819242477417, -11.967722237110138, -3.7717437148094177]", "[1.4640089869499207, -11.975127398967743, -3.8629953265190125]", "[-0.3047674596309662, -0.008611917495727539, 0.4807789921760559]", "[-0.43397413194179535, -0.08774667978286743, 0.4424424171447754]", "[-0.15135809779167175, -0.5899825096130371, 0.038314998149871826]", "[-0.14424720406532288, -0.6127802133560181, 0.00409466028213501]", "[-0.15821045637130737, -0.004229068756103516, 0.1018596887588501]", "[-0.16024360060691833, -0.013019323348999023, 0.08916616439819336]", "[0.023436278104782104, -0.00023919343948364258, 0.10835212469100952]", "[0.003017321228981018, -0.004502594470977783, 0.10847455263137817]", "[-0.03104814887046814, -0.037895798683166504, 0.12315988540649414]", "[-0.03161226212978363, -0.038371384143829346, 0.12258356809616089]", "[-0.03439028561115265, -0.03818875551223755, 0.12368619441986084]", "[-0.037670403718948364, -0.037632644176483154, 0.12549656629562378]", "[-0.038258060812950134, -0.038200974464416504, 0.12477880716323853]"
#
# # In[ ]:
#
#
# t2 = "[0.07289934158325195, -0.2344406843185425, -0.1477822288870811]", "[0.2312825322151184, -0.05772793292999268, -0.3205077052116394]", "[0.2806476354598999, -0.01441717147827148, -0.2496735155582428]", "[0.3066678047180176, 0.05454879999160767, -0.3417018353939056]", "[0.1782215237617493, -0.04886579513549805, -0.5152961015701294]", "[-0.002878665924072266, -0.7520697712898254, -0.960427850484848]", "[-0.5670732855796814, -0.555764377117157, -0.8987250812351704]", "[1.240435302257538, 3.454282879829407, 1.116443760693073]", "[-0.129380851984024, 0.927283763885498, -0.2805608315393329]", "[-2.385889202356339, -3.803887844085693, -2.633329410105944]", "[0.3237326443195343, 1.436899602413177, 0.3644149005413055]", "[1.498748563230038, 4.410494208335876, 0.9024870246648788]", "[-0.9180435240268707, 0.4196292757987976, -0.8116500601172447]", "[-2.098156064748764, -5.647458910942078, -2.149242119863629]", "[0.471931267529726, 2.493711054325104, 0.7828115094453096]", "[1.491843761876225, 5.303093194961548, 1.190858401358128]", "[-4.108473360538483, -2.935313880443573, -2.292653787881136]", "[-1.795808762311935, -3.026154279708862, -1.448967903852463]", "[0.5379805192351341, 5.965560495853424, 1.69856833666563]", "[0.7695640325546265, 4.798599123954773, 1.048382957698777]"
#
# # In[ ]:
#
#
# temp = []
# t20 = []
# for i in t2:
#     t3 = i[1:-1].split(',')
#     if (len(t3) == 3):
#         t3_num = [float(x) for x in t3]
#         t20.append(t3_num)
# temp.append(t20)
# prepare_data = np.array(temp)
# prepare_data.shape
#
# # In[ ]:
#
#
# result = model.predict(prepare_data)
# result
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# aaa = [1, 2, 3]
#
# # In[ ]:
#
#
# aaa.index(1)
#
# # In[ ]:
#
#
# from PIL import Image
#
# img = Image.open('uploads/client0.png')
# img.size
#
# # In[ ]:
#
#
# img.thumbnail((64, 64), Image.ANTIALIAS)
#
# # In[ ]:
#
#
# img.save('uploads/simpson.png')
#
# # In[ ]:
#
#
# if os.path.exists("uploads/simpson.png"):
#
#
