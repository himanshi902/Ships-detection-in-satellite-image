
# coding: utf-8

# In[1]:


import json, sys, random
import numpy as np


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D , MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks


# In[3]:



import PIL
from PIL import Image, ImageDraw


# In[4]:


from matplotlib import pyplot as plt


# In[5]:


# Download and study the dataset
f = open(r'Desktop/major project/shipsnet.json')
import json
dataset = json.load(f)
f.close()


# In[7]:



input_data=np.array(dataset['data']).astype('uint8')
output_data=np.array(dataset['labels']).astype('uint8')


# In[8]:


input_data.shape


# In[9]:


input_data


# In[ ]:


# to be able to read image we need to reshape the array or input_data


# In[10]:


n_spectrum=3
weight=80
height=80
X=input_data.reshape([-1,n_spectrum,weight,height])
X[0].shape


# In[11]:


#get one channel
pic=X[3]

red_spectrum=pic[0]
green_spectrum=pic[1]
blue_spectrum=pic[2]


# In[12]:


plt.figure(2,figsize=(5*3,5*1))
plt.set_cmap('jet')

plt.subplot(1,3,1)
plt.imshow(red_spectrum)

plt.subplot(1,3,2)
plt.imshow(green_spectrum)

plt.subplot(1,3,3)
plt.imshow(blue_spectrum)

plt.show()


# In[13]:


output_data


# In[14]:


np.bincount(output_data)


# In[15]:


# output encoding
y=np_utils.to_categorical(output_data,2)


# In[17]:


indexes=np.arange(4000)
np.random.shuffle(indexes)


# In[18]:


X_train=X[indexes].transpose([0,2,3,1])
y_train=y[indexes]


# In[19]:


X_train=X_train / 255 # to have value between 0 and 1 


# In[20]:


# train the network
np.random.seed(42)


# In[21]:



# network design
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))


# In[22]:



# optimization setup
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

# training
model.fit(
    X_train, 
    y_train,
    batch_size=32,
    epochs=18,
    validation_split=0.2,
    shuffle=True,
    verbose=2)


# In[23]:


# batch size = 32 photos at once

# Using Network

# download image
image = Image.open(r'Desktop/major project/scenes/sfbay_1.png')
pix = image.load()


# In[24]:


plt.imshow(image)


# In[25]:


n_spectrum = 3
width = image.size[0]
height = image.size[1]


# In[26]:


# creat vector
picture_vector = []
for chanel in range(n_spectrum):
    for y in range(height):
        for x in range(width):
            picture_vector.append(pix[x, y][chanel])


# In[27]:


picture_vector = np.array(picture_vector).astype('uint8')
picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)


# In[28]:


plt.figure(1, figsize = (15, 30))

plt.subplot(3, 1, 1)
plt.imshow(picture_tensor)

plt.show()


# In[29]:


picture_tensor = picture_tensor.transpose(2,0,1)


# In[30]:


# Search on the image

def cutting(x, y):
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture_tensor[0][y+i][x+j]
            area_study[1][i][j] = picture_tensor[1][y+i][x+j]
            area_study[2][i][j] = picture_tensor[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study


# In[31]:


def not_near(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result


# In[32]:


def show_ship(x, y, acc, thickness=5):   
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+th+80][x+i] = -1


# In[33]:


step = 10; coordinates = []
for y in range(int((height-(80-step))/step)):
    for x in range(int((width-(80-step))/step) ):
        area = cutting(x*step, y*step)
        result = model.predict(area)
        if result[0][1] > 0.90 and not_near(x*step,y*step, 88, coordinates):
            coordinates.append([[x*step, y*step], result])
            print(result)
            plt.imshow(area[0])
            plt.show()


# In[34]:


for e in coordinates:
    show_ship(e[0][0], e[0][1], e[1][0][1])


# In[38]:


picture_tensor = picture_tensor.transpose(2,0,1)
picture_tensor.shape


# In[39]:



plt.figure(1, figsize = (15, 30))

plt.subplot(3,1,1)
plt.imshow(picture_tensor)

plt.show()

