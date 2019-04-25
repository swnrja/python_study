import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
import seaborn as sns
import os
import subprocess
import sys
from sklearn.preprocessing import MinMaxScaler

t_featurename__  = ['C', 'Cr', 'Mo', 'Co', 'Ni', 'Nb', 'W', 'Mn', 'Si', 'V', 'aus', 'tem', 'time', 'AF', 'DQ']


def help():
   print("\t- HELP")
   print("\t- 1. statistics.py: \t  통계 프로그램")
   print("\t- 2. tf_BIGDATA.py: \t  머신 러닝 프로그램")
   print("\t- 2. 실행 방법: \t  python tf_BIGDATA.py -i(input) TARGETDB.csv -T(target) target")   
   print("\t- 3. ga.py -i(input) targetDB.csv: \t\t  유전 알고리즘 프로그램")
   print("\t- 3. genetic algorithm")
   print("\t- you can eash step's help option in the each file")

#1. Dataload and split data
def create_model():
   try: #DB입력하지 않았을 때
      input_data = sys.argv[2]   
   except IndexError:
      print("DB Syntax and target syntax required or check the syntax, goto help(h)")
      exit(0)

   try: #DB가 없을 때
      data = pd.read_csv(input_data) 
   except FileNotFoundError:
      print("Exact DB Input Required or check the syntax, goto help(h)")
      exit(0)

   try: #-T를 입력하지 않았을 때
      target_data = sys.argv[3] # -T
   except IndexError:
      print("Target syntax required or check the syntax, goto help(h)")
      exit(0)   
   
   try: #target을 잘못 입력했을 때
      target_data = sys.argv[4] # UTS ? YS ? ...?
   except IndexError:
      print("Target syntax required or check the syntax, goto help(h)")
      exit(0)


########

   data = data.dropna()
   df1 = pd.read_csv(input_data)
   
   try: #target을 입력하지 않았을 때. 
      y = data.loc[:, [target_data]].values
   except KeyError:
      print("Exact target value or check the syntax, goto HELP(h)")
      exit(0)
   
   
   x = data.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]].values 
   
   command_i = ['-i','-input'] #input 실행
   if sys.argv[1] not in command_i:
      print("-i or input is required, goto help(h)")
      exit(0)

   command_t = ['-T','-target'] #target 실행
   if sys.argv[3] not in command_t:
      print("-T or target is required, goto help(h)")
      exit(0)


   
   #-----------------------------------split data to test, train, validation-------------------------------------------------#
   X_train_total, X_test, y_train_total, y_test = train_test_split(x, y, random_state=0)
   X_train, X_val, y_train, y_val = train_test_split(X_train_total, y_train_total, random_state = 0)
   #------------------------------------------------------------------------------------#
   #1.1 Scaling - using standard scaler
   Scaler = StandardScaler()
   Scaler.fit(X_train)
   X_train = Scaler.transform(X_train)
   X_val = Scaler.transform(X_val)
   X_test = Scaler.transform(X_test)

   keras.initializers.he_uniform(seed=None) # He initizlier - Relu is addaptive. 
   #2. modeling
   a = 64
   b = 64
   
   
   model = Sequential()
   model.add(Dense(a,activation='relu',))
   model.add(Dense(b,activation='relu',))


   model.add(Dense(1))

   X_train = np.array(X_train)
   y_train = np.array(y_train)

   #################################################

   #RMSE
   def root_mean_squared_error(y_true,y_pred):
        from keras import backend as K
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

   #R2
   def R_Squared(y_true, y_pred):
      from keras import backend as K
      SS_res =  K.sum(K.square( y_true-y_pred ))
      SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
      return ( 1 - SS_res/(SS_tot + K.epsilon()) )

   #################################################  

   #Complie model
   #model.compile(loss = root_mean_squared_error, optimizer= Adam(lr=0.0005), metrics=[root_mean_squared_error,R_Squared])
   model.compile(loss = 'mean_absolute_error', optimizer= Adam(lr=0.0005), metrics=['mae',R_Squared]) 
   history = model.fit(X_train, y_train, epochs=300, batch_size=5, validation_data=(X_val, y_val))
   model.save('model_'+target_data+'.h5') # file이름에 targe붙이기
   #load_model('model.h5', custom_objects={"R_Squared": R_squared}


   #3. plotting 
   # 3.1. loss and MAE
   while True:
            print("\n")
            namel = input("1. Loss Graph and NAN -l help -h and next -n\n")
            print("\n")
            if namel == '-n':
               break
            #elif namel == '-m':

            #   plt.plot(history.history['mean_absolute_error'],c='b', label = 'train_mae')
            #   plt.plot(history.history['val_mean_absolute_error'], c='r', label = 'validation_mae')
            #   plt.xlabel('epoch')
            #   plt.ylabel('mae')
            #   plt.legend(loc='upper left')
            #   plt.show()
            #   model.evaluate(X_test, y_test)
            elif namel =='-l':

               #값을 쓰고 


               #loss_to_csv
               np.savetxt(target_data+"_Train_Loss.csv", history.history['mean_absolute_error'], delimiter=",")
               np.savetxt(target_data+"_Test_Loss.csv", history.history['val_mean_absolute_error'], delimiter=",")
               
               plt.plot(history.history['loss'],c='r', label = 'train loss')
               plt.plot(history.history['val_loss'], 'b', label = 'test loss')
               plt.xlabel('epoch')
               plt.ylabel('loss')
               plt.legend(loc='upper left')
               plt.show()
               




            elif namel =='-h':
               print("-h\tIf you want to see the help file\n")
               print("-n\tIf you want to go to next and see Train and Test graph.\n")
               #print("-m\tto see the mae graph.\n")
               print("-l\tto see the loss graph.\n")
               

   #3.2 
   # train set predict.
   # train and test
   while True:
            print("\n")
            names = input("2. TRain graph -TR and TesT graph -TT help -h and Quit -q\n")
            print("\n")
            if names == '-q':
               break
            elif names =='-TR':
               real_value_tr_x = X_train
               predict_tr = model.predict(X_train)
               real_value_tr = y_train
################## csv로 
               result_train_csv = pd.DataFrame({'real_value_tr':list(real_value_tr.T[0]),    'predict_tr':list(predict_tr.T[0])},)
               name = pd.DataFrame(X_train)
               name.columns = t_featurename__
               result_train_csv = pd.concat([name,result_train_csv],axis=1)
               result_train_csv.to_csv(target_data+"_Train_Predict"+".csv",header = True, index = False) # To .csv_Train
               
################## matplotlib로 그리는 것
               result1 = pd.DataFrame({'real_value_tr':list(real_value_tr.T[0]),    'predict_tr':list(predict_tr)},)
               result1 = result1.sort_values(by='predict_tr')
               result1.index = range(len(result1))
               fig, axes = plt.subplots(figsize = (15,10))
               plt.plot(result1['predict_tr'], label = 'Prediction')
               plt.plot(result1['real_value_tr'], 'o', label = 'Real Value')
               plt.ylabel('UTS')
               plt.xlabel('index')
               plt.legend(loc = 'best')
               plt.show()
               
            
   # 3.2. test set predict
               
            elif names =='-TT':

               real_value_tt_x = X_test
               predict_tt = model.predict(X_test)
               real_value_tt = y_test
################## X축 Predict, y축 real
               result_test_csv = pd.DataFrame({'real_value_tt':list(real_value_tt.T[0]),    'predict_tt':list(predict_tt.T[0])},)
               name = pd.DataFrame(X_test)
               name.columns = t_featurename__
               result_test_csv = pd.concat([name,result_test_csv],axis=1)
               result_test_csv.to_csv(target_data+"_Test_Predict"+".csv",header = True, index = False) # To .csv_Test
################## 
               result1 = pd.DataFrame({'real_value_tt':list(real_value_tt.T[0]),    'predict_tt':list(predict_tt)},) #
               result1 = result1.sort_values(by='predict_tt') #Graph for ascending order
               result1.index = range(len(result1)) # index length
               fig, axes = plt.subplots(figsize = (15,10))
               plt.plot(result1['predict_tt'], label = 'Prediction')
               plt.plot(result1['real_value_tt'], 'o', label = 'Real Value')
               plt.ylabel('UTS')
               plt.xlabel('index')
               plt.legend(loc = 'best')
               plt.show()
              
         
            elif names =='-h':
               print("-q\tQuit this file..\n")
               print("-TR\tto see the Train Y^ graph.\n")
               print("-TT\tto see the Test Y^ graph.\n")    
               print("-TT\tTo see the help file.\n")      
   # 4.print R2 and MAE

   z = model.evaluate(X_test, y_test)
   x = model.evaluate(X_train, y_train)
   
   print("MAE of train: ", x[1])
   print("R2 of train: ", x[2])
 
   print("MAE of test: ", z[1])
   print("R2 of test: ", z[2])
   #pearson correlation...?

   
   return

  
                             

 
create_model()
