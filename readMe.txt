enviroment setup

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


project structure
