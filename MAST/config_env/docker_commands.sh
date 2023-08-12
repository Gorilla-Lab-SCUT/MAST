conda activate base
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
cd /root
mkdir deps && cd deps
git clone https://github.com/ylabbe/job-runner.git
git clone https://github.com/ylabbe/bullet3.git
git clone https://github.com/ylabbe/bop_toolkit_cosypose.git
git clone https://github.com/ylabbe/bop_toolkit_challenge20.git
cd ..
pip install --upgrade pip setuptools
pip install --ignore-installed PyYAML==5.1
pip install --no-deps e3nn
conda install -c anaconda gxx_linux-64
conda install -c conda-forge/label/cf202003 pinocchio
conda install ffmpeg rclone bzip2
conda install -c conda-forge ninja
python setup.py install
pip install PyOpenGL triangle glumpy pypng trimesh xarray pyarrow gpustat simplejson
apt-get install -y libfreetype6 libfreetype6-dev libglfw3 sshfs

# some library need by pinocchio such as "console_bridge", "assimp4.1.0", search and install via conda
