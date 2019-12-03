# 머신러닝및 딥러닝. 
[text_analysis.ipynb](text_analysis.ipynb) 소설 동백꽃에 Word2Vec를 적용.  
[carry on x plus y.ipynb](carry%20on%20x%20plus%20y.ipynb) x + y 에서 자리올림이 발생하는 경우를 SVC로 학습.  
## 텐서플로우란?
딥러닝을 통해서 사람이 로직을 넣지 않아도 데이터를 통해서 자동으로 로직이 있는거 같이 동작.  
[in Colaboratory](https://colab.research.google.com/drive/1Sv3lAAXJy55tnFL0MT9bGmYRO5MilWQ2)  
# 설정
## KoNLPy 설치
아나콘다 프롬프트를 실행시켜서 KoNLPy에 필요한 패키지를 설치한다.  
프롬프트에서 오류가 나오면 관리자 프롬프트에서 설치하면됨.  
conda로 설치가 안되면 pip로 설치를 하면 됨.  
```
conda install openjdk
pip install --upgrade pip
pip install jpype1
pip install konlpy
```
분명히 OpenJDK를 설치를 했는데 경로가 지정이 안된다고 나올경우에 경로 확인후 실행.
```py
import os
if 'JAVA_HOME' not in os.environ:
    os.environ['JAVA_HOME']=r'C:\Users\q\Anaconda3\Library'
```
## tensorflow-gpu cuda폴더 설정
cuda와 cudnn의 동적 라이브러리 파일이 있는 경로를 추가한다.  
```py
import os
import pathlib
if 'cudnn' not in os.environ['PATH']:
    os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudnn-7.6.4-cuda10.0_0\Library\include'))+os.environ['PATH']
    os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudnn-7.6.4-cuda10.0_0\Library\bin'))+os.environ['PATH']
    os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudnn-7.6.4-cuda10.0_0\Library\lib\x64'))+os.environ['PATH']
    os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudatoolkit-10.0.130-0\DLLs'))+os.environ['PATH']
    os.environ['PATH']=str(pathlib.PurePath(r'C:\Users\q\Anaconda3\pkgs\cudatoolkit-10.0.130-0\Library\bin'))+os.environ['PATH']
```
## Failed to get convolution algorithm.
Tensorflow에서 GPU사용시 Failed to get convolution algorithm. 오류가 나올때  
```py
#https://stackoverflow.com/a/59007505 : Failed to get convolution algorithm. 오류
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
```