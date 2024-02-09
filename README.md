# CatDetection
Original cat detection by haar cascade was not even used in final version.
This project evolves into cat recognition system. It is taillored to my needs - it sends pushbullet notification when specific cat is detected in video feed from netweork camera. Reason for this project is that our ginger garfield needs special diet, and he should not eat together with other cats. But he is very clever and sometimes he get to other cats food. thanks to these python script I get notification when is is happen.  

There is database of pictures (8 for each cat, 4 in color, 4 in BW nigth vision mode) and precomputed vectors for each image. 
Then there is main scipt CatAlarmSystem.py which get video from camera, do motion detection, if some movement is detected then AI model analyse image and find best matching image in database to identify specific cat.

## enviroment setup

install python3
install miniconda - python enviroment, miniconda instead of full anaconda download packeges as needed, it does not download everything during installation.

sources - setup directML tensor on windows
https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows

miniconda download https://docs.conda.io/projects/miniconda/en/latest/

miniconda setup
conda create --name pydml -y
conda activate pydml

another enviroment setup and pip installs
conda install numpy pandas tensorboard matplotlib tqdm pyyaml -y
pip install opencv-python
pip install wget
pip install torchvision
conda install pytorch cpuonly -c pytorch
pip install torch-directml
pip install numpy

pip install pushbullet-python
pip install python-dotenv

### modules


## Result
![](https://github.com/skopy123/CatDetection-HaarCascade/blob/master/testImg/detectionResulst.jpg)




