# import packages needed
import sklearn
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# load digits from the sklearn.datasets
digits = load_digits()

# x_data means all data matrix value [0 1]
x_data_all = digits.data

# y_data means all data real value [1797]
y_data_all = digits.target


# print how many pixel matrixes we have
print(x_data_all.shape)
print(" ------------------ ")
# print how many data real value we have
print(y_data_all.shape)
print(" ------------------ ")

# show the first image
def show_single_img(img_arr):
    plt.imshow(img_arr, cmap='gray')
    plt.show()
show_single_img(digits.images[0])

# print pixel matrix with index 0
print(digits.images[0])
print(" ------------------ ")
print(x_data_all[0])
print(" ------------------ ")
# print target data value with index 0
print(y_data_all[0])

print(" ------------------ ")

# test print index=457 element, try to convince that different
# index element will have different value from 0 - 9
print(y_data_all[456])
print(" ------------------ ")

# the random_state parameter is used for initializing the internal random number
# generator, which will decide the splitting of data into train and test indices in case
# divide dataset first time (train all, test)
x_train_all, x_test, y_train_all, y_test = train_test_split(
    x_data_all, y_data_all, random_state=7
)

# divide dataset second time  (train, valid)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)

# print to show the divide train and test datasets
print(x_data_all.shape, y_data_all.shape)
print(x_train_all.shape, y_train_all.shape)
print(x_test.shape, y_test.shape)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)

# prevent one column too big, do the standard normalization
# we can understand as standardize features by removing the mean and scaling unit variance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)



#Define the network structure:
model = keras.models.Sequential()  #Create an empty sequential model

#Add an input layer:
model.add(keras.layers.Dense(30, input_shape = x_train.shape[1:])) #The input shape is 1*64

#deep network structure: hidden layers:
for _ in range(20):
    model.add(keras.layers.Dense(100, activation = 'relu')) 
#Rectified Linear Unit:if input>0, return input itself, if input<=0, return 0

#set dropout to prevent overfitting:
model.add(keras.layers.AlphaDropout(rate = 0.5))

#add output layer:
model.add(keras.layers.Dense(10, activation = 'softmax')) 
#The output is a real number between 0.0 and 1.0
#And the sum of the output values is always 1.0

#Specify loss functions, optimization methods, and performance metrics
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics=["accuracy"])
#loss='sparse_categorical_crossentropy' specifies cross entropy as loss function

#check hidden layers
model.summary()

#model training
history = model.fit(x_train_scaled, y_train, epochs= 200,
                    validation_data= (x_valid_scaled, y_valid))

#draw picture to show loss and accucacy
def plt_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize= (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,4)
    plt.show()
plt_learning_curve(history)

#only to see accuracy
def plt_acc_curve(history):
    acc = history.history['accuracy'] 
    val_acc = history.history['val_accuracy'] 
    epochs = range(1,len(acc)+1)
    plt.plot(epochs,acc,'orange',label='accurracy')
    plt.plot(epochs,val_acc,'red',label='val_accuracy')
    plt.grid(True)
    plt.legend() 
    plt.show()
plt_acc_curve(history)

# Evaluate each category,show probability
print("[INFO] evaluating network...")
predictions = model.predict(x_test_scaled)
for i in range (300,310):
    print(predictions[i])
print(predictions.shape)

#to see each prediction of y
predictions2 = model.predict(x_test_scaled)
y_test_pred = np.argmax(predictions2, axis=1)
print(y_test_pred)