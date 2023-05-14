#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
from glob import glob
from sklearn.metrics import roc_curve, auc
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_auc_score
from tensorflow.keras import callbacks 
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from  matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,BatchNormalization,Dropout
import shutil
from sklearn.metrics import  precision_score, recall_score, accuracy_score,classification_report ,confusion_matrix
from tensorflow.python.platform import build_info as tf_build_info
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


# In[6]:


Diagnosis = {
    4 : "MEL: Melanoma\n Melanoma is a type of skin cancer that occurs in the melanocyte cells. It is the most dangerous form of skin cancer and can spread to other parts of the body. It usually appears as a dark, irregularly shaped mole with uneven coloration. \n Treatment involves surgical removal of the affected skin, followed by radiation therapy or chemotherapy. Early detection and treatment is crucial for a better prognosis.",
    5: "NV: Melanocytic Nevi \n Melanocytic nevi, also known as moles, are benign growths that occur when melanocytes grow in a cluster. They are usually round or oval in shape, and can be brown, black, or skin-colored. \n Treatment is not necessary unless the mole changes in size, color, or shape, in which case it should be removed and biopsied to rule out cancer.",
    1: "BCC: Basal Cell Carcinoma \n Basal cell carcinoma is a type of skin cancer that begins in the basal cells, which produce new skin cells. It usually appears as a small, shiny bump or a pink patch of skin, and can sometimes bleed or develop a crust. \n Treatment involves surgical removal of the affected skin, often with the use of Mohs surgery, a procedure that removes cancerous tissue layer by layer until all of the cancer has been removed.",
    0: "AKIEC: Actinic Keratoses \n Actinic keratoses are precancerous growths that occur on sun-exposed areas of the skin, such as the face, scalp, and hands. They usually appear as dry, scaly patches or bumps, and can be red, pink, or skin-colored. \n Treatment involves freezing the growths with liquid nitrogen, using medicated creams, or removing them surgically.",
    2: "BKL: Benign Keratosis-like Lesions \n Benign keratosis-like lesions are benign growths that can resemble actinic keratoses, but are not precancerous. They usually appear as scaly, rough patches or plaques, and can be pink, red, or brown. \n Treatment is not necessary unless the lesion changes in size, color, or shape, in which case it should be removed and biopsied to rule out cancer.",
    3: "DF: Dermatofibroma \n Dermatofibromas are benign growths that occur in the dermis, the layer of skin beneath the epidermis. They usually appear as small, firm bumps that are brown, pink, or red, and can be itchy or painful.\n Treatment is not necessary unless the growth becomes bothersome, in which case it can be removed surgically.",
    6: "VASC: Vascular Lesions \n Vascular lesions are growths that occur in the blood vessels of the skin. They can be either benign or malignant, and include conditions such as hemangiomas and angiokeratomas. \n Treatment depends on the type of lesion and may involve surgical removal, laser therapy, or medication.",
}


# In[7]:


from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
import tensorflow as tf



class SoftAttention(Layer):
    def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        
        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
    
        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
        
        if self.aggregate_channels==False:

            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape
        

        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='he_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):

        exp_x = K.expand_dims(x,axis=-1)

        c3d = K.conv3d(exp_x,
                     kernel=self.kernel_conv3d,
                     strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                        self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)

        conv3d = K.permute_dimensions(conv3d,pattern=(0,4,1,2,3))

        
        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))

        softmax_alpha = K.softmax(conv3d, axis=-1) 
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1],self.i_shape[2]))(softmax_alpha)

        
        if self.aggregate_channels==False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)       
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))
   
            x_exp = K.expand_dims(x,axis=-2)
   
            u = kl.Multiply()([exp_softmax_alpha, x_exp])   
  
            u = kl.Reshape(target_shape=(self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))

            exp_softmax_alpha = K.sum(exp_softmax_alpha,axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])   

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u,x])
        else:
            o = u
        
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape): 
        return [self.out_features_shape, self.out_attention_maps_shape]

    
    def get_config(self):
        return super(SoftAttention,self).get_config()
en = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,

)
# Exclude the last 28 layers of the model.
conv = en.output
attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
attention_layer=(MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
conv=(MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

conv = concatenate([conv,attention_layer])
conv  = Activation('relu')(conv)
conv = Dropout(0.25)(conv)
output = Flatten()(conv)
output = Dense(7, activation='softmax')(output)
model = Model(inputs=en.input, outputs=output)
opt1=tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=0.1)
model.compile(optimizer=opt1,
             loss='categorical_crossentropy',
             metrics=['accuracy'])
from tensorflow.keras import models
model.load_weights("e_4a5.h5")


# In[14]:


import matplotlib.pyplot as plt
import cv2
files=os.listdir('test')


# In[17]:


for file_name in files:
    test=plt.imread(r'test/'+file_name)
    test=cv2.resize(test,(224,224))
    test=np.reshape(test,(1,224,224,3))
    result=model.predict(test,verbose=0)
    print()
    print()
    print()
    print()
    print('Predicted Classes of test image:',file_name,Diagnosis[np.argmax(result)])
    print()
    print()
    
    


# In[ ]:




