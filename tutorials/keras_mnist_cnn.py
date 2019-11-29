"""
오리지널 주소 : https://www.tensorflow.org/tutorials/images/cnn
path지정하는곳 을 먼저 실행함.
https://stackoverflow.com/a/59007505 에 나온 Failed to get convolution algorithm. 오류 해결 소스를 추가함.

windows 10, anaconda환경에서 작업.
cudatoolkit : 10.0.130
cudnn : 7.6.4
tensorflow : 2.0.0
tensorflow-gpu : 2.0.0

gpu : GTX 960
"""

"""
# path가 지정이 되어있지 않으면 cuda관련 dll을 찾지 못하기 때문에 지정을 해 주어야 한다.
import os
import pathlib
ospath=os.environ['PATH']
os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudnn-7.6.4-cuda10.0_0\Library\include'))+os.environ['PATH']
os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudnn-7.6.4-cuda10.0_0\Library\bin'))+os.environ['PATH']
os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudnn-7.6.4-cuda10.0_0\Library\lib\x64'))+os.environ['PATH']
os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudatoolkit-10.0.130-0\DLLs'))+os.environ['PATH']
os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudatoolkit-10.0.130-0\Library\bin'))    +os.environ['PATH']
print(os.environ['PATH'])
"""

#https://www.tensorflow.org/tutorials/images/cnn
from __future__ import absolute_import, division, print_function, unicode_literals

#!pip install -q tensorflow-gpu==2.0.0-rc1
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0


#https://stackoverflow.com/a/59007505 : Failed to get convolution algorithm. 오류
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

#tf.config.gpu.set_per_process_memory_growth(True)
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#합성곱층 만들기
"""
cnn 입력 : tensor : 이미지 눂이, 너비, 컬러
"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary() #모델의 구조

#dense 층 추가.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

#모델 컴파일과 훈련하기
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

#모델 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)