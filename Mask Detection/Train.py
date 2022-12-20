#!/usr/bin/env python
# coding: utf-8

# # 20/12/2022
# 
# # Deep Learning va Computer Vision 
# 
# # Mask Detection
# 
# # Muallif: Farrux Sotivoldiyev

# #### `Kutubxonalar`

# In[1]:


import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.applications.mobilenet import preprocess_input


# #### `Google Drive va Google Colab ni bog'lab olish`

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# #### `Rasmlar turgan path`

# In[3]:


train_data_dir = "/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/Train"
test_data_dir = "/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/Test"


# #### `Path ni olib undan rasmlarni o'qib (224,224) o'lchamga olib kelib x_train,y_train ga bo'lib qaytaradi` 

# In[1]:


def get_load_dataset(data_dir):
    data = []
    x = []
    y = []
    classes = ["mask","no_mask"]
    for i in classes:
        path = os.path.join(data_dir,i)
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img))
            img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array,(224,224))
            encod = 0 if i=="mask" else 1 # mask = 0 ; no_mask = 1
            if img_array.shape[2]==3:
                data.append((img_array,encod))
    np.random.shuffle(data)
    for i,j in data:
        x.append(i)
        y.append(j)
    x = np.array(x, dtype= np.float32).reshape(-1, 224, 224,3)
    y = np.array(y, dtype=np.uint8)
    return preprocess_input(x), y


# #### `Data x_train,y_train ga yuklandi`

# In[5]:


x_train,y_train = get_load_dataset(train_data_dir)


# In[6]:


x_test,y_test = get_load_dataset(test_data_dir)


# #### x_train, y_train va  x_test, y_test uzunligini ko'rish

# In[7]:


print("x_train len:",len(x_train))
print("y_train len:",len(y_train))
print("x_test len:",len(x_test))
print("y_test len:",len(y_test))


# #### x_train, y_train va  x_test, y_test shape ni ko'rish

# In[8]:


print("x_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)
print("x_test shape:",x_test.shape)
print("y_test shape:",y_test.shape)


# # Training

# #### `Training uchun MobileNet applicationini yuklab olish`

# In[9]:


model_app = tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3),include_top=False)


# #### `Modelning arxitekturasini ko'ramiz`

# In[10]:


# model_app.summary()


# #### `Modelimiz Transfer Learning bo'lishi uchun model ichidagi barcha layerlarni muzlatib qo'yishimiz kerak`

# In[11]:


model_app.trainable = False


# #### `Tuzayotgan clasificatsiyamizga qarab Transfer learning bo'lgan modelimizga qo'shimcha layer qo'shamiz`

# In[13]:


model = Sequential([
    model_app,
    Flatten(),
    Dense(1000, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid"),
])


# #### `Classifikatsiyamizga mos model arxitekturasini ko'ramiz`

# In[14]:


# model.summary()


# #### `Modelni compile qilish: bunda optimizer,loss va metrics ni berishimiz kerak` 

# In[15]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# #### `Modelga bilim berish`

# In[16]:


model.fit(x_train,y_train,epochs=5,validation_split=0.3)


# #### `Modelni parameterlarini saqlash ya'ni weight va biaslarini`

# In[28]:


model.save("/content/drive/MyDrive/Colab Notebooks/train_new_models/train_mask_detection2.h5")


# #### `Modelni chaqirib olish`

# In[29]:


new_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/train_new_models/train_mask_detection2.h5")


# # Tekshirinb ko'ramiz

# #### `Maskasiz`

# In[30]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/farruxbek_2.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[31]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/farruxbek_1.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[32]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/person1.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[33]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/person2.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[34]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/215.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# #### `Maskali`

# In[35]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/mask1.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[36]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/mask2.jpg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[37]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/with_mask620.jpeg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# In[38]:


rasm = plt.imread("/content/drive/MyDrive/Colab Notebooks/datasets/Face_Mask_Dataset/rasmlar/with_mask624.jpeg")
rasm = cv2.resize(rasm,(224,224))
rasm = np.expand_dims(rasm,0)
rasm = rasm / 255.0
new_model.predict(rasm)


# #### `Xulosa: Model bashoratiga ko'ra Maskali rasmlar 0.5 dan kichik, Maskasiz esa 0.5 dan katta bo'ladi`
