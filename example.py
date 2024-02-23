from numpy.lib import math
from IPython.core.display import Math
from tensorflow.python.data.ops.options import options_lib
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


optizer_func = [
    'SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam' ];

actFuncNames = ['ReLU',
     'LeakyReLu', 'Mish', 'Swish', 'Smish', 'Logish','Softplus','Proposed'
    ];
 
 
def actFuncs(x):
  switcher = {
      0 : tf.maximum(0.0,x),
      1 : tf.maximum(0.9 * x, x),                                # LeakyReLu
      2 : x * K.tanh(K.log(1.0 + K.exp(x))),                     # Mish
      3 : x * (1.0 / (1.0 + K.exp(-x))),                         # Swish
      4 : x * K.tanh(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),    # Smish
      5 : x * K.log(1.0 + (1.0 / (1.0 + K.exp(-x)))),            # Logish
      6:  K.softplus(x),
      7:   x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      #8:  x * K.sigmoid( K.log ( 1.0 +  K.tanh(x) ) )+0.1,
      #tf.where(x > 0.0,  x*K.tanh(K.log(1.0 + (1.0 / (1.0 + K.exp(-x)))))+x ,0.9 * x)
     
     
      }
  return switcher.get(funcID, "Invalid function ID!");

"""actFuncNames = ['Proposed_0.1','Proposed_0.2','Proposed_0.3','Proposed_0.4','Proposed_0.5',
                'Proposed_0.6','Proposed_0.7','Proposed_0.8','Proposed_0.9'
    ];
 
 
def actFuncs(x):
  switcher = {
      0 : 0.1*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      1 : 0.2*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),                                # trish
      2 : 0.3*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      3 : 0.4*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      4 : 0.5*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      5 : 0.6*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      6:  0.7*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      7:   0.8*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),
      8:   0.9*x * K.sigmoid(K.log(1.0 + (1.0 / (1.0 + K.exp(-x))))),

      #tf.where(x > 0.0,  x*K.tanh(K.log(1.0 + (1.0 / (1.0 + K.exp(-x)))))+x ,0.9 * x)
     
     
      }
  return switcher.get(funcID, "Invalid function ID!");"""

from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization,Lambda
def deepModel(x):
  opt=x
  cnnmodel=Sequential()
  #cnnmodel.add(Conv2D(8,(3,3),input_shape= (224, 224, 3)))
  cnnmodel.add(Conv2D(8,(3,3),input_shape=x_train.shape[1:]))
  cnnmodel.add(MaxPooling2D((2,2)))
  cnnmodel.add(Conv2D(32,(3,3),activation=actFuncs))
  cnnmodel.add(MaxPooling2D((2,2)))
  cnnmodel.add(Conv2D(32,(3,3),activation=actFuncs))
  cnnmodel.add(MaxPooling2D((2,2)))
  cnnmodel.add(BatchNormalization())

  cnnmodel.add(Flatten())
  cnnmodel.add(Dense(128,activation=actFuncs))
  cnnmodel.add(Dense(10,activation='softmax'))



  cnnmodel.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy',f1_m])
  res_cnnmodel=cnnmodel.fit(x_train, y_train, validation_data = (x_test, y_test),epochs=50)
  #plot(res_cnnmodel)

  # Test model
  return cnnmodel.evaluate(x_test, y_test, verbose = 1)



# iterate over the list using index
for j in range(len(optizer_func)):
  print(j + 7, ". Optimizer Function: ", optizer_func[j+6])
  for i in range(len(actFuncNames)):
    print(i + 1, ". Activation Function: ", actFuncNames[i])
    print('---------------------------------------------------------------------')

    funcID = i; # Burası aktivasyon fonksiyonunu otomatik olarak seçmek için kullanıldı!

    #Belirlenen parametreler ve eğitim veri seti ile kurulan modelin eğitilmesi  
    scores = deepModel(optizer_func[j+6])
    print(' ')
    print(' ')
