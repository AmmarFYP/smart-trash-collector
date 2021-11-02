#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


fashiondata = tf.keras.datasets.mnist


# In[3]:


(X_train, Y_train),(X_test, Y_test) = fashiondata.load_data()


# In[4]:


X_train.shape


# In[5]:


X_train, X_test = X_train/255, X_test/255


# In[6]:


model = tf.keras.models.Sequential([
    tf.keras.layer.Flatten(input_shape = (28,28)),
    tf.keras.layer.Dense(128, activation ='relu'),
    tf.keras.layer.Dropout(0.2),
    tf.keras.layer.Dense(10, activation = 'softmax')
])


# In[7]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[8]:


model.fit(X_train, Y_train, epochs=5)


# In[9]:


model.evaluate(X_test, Y_test)


# In[ ]:




