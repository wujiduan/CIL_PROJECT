module load gcc/6.3.0 python_gpu/3.7.1 cuda/10.1.243 cudnn/7.6.4 eth_proxy
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip3 install -r requirements_leonhard.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
export PYTHONPATH=$PYTHONPATH:~/CIL_PROJECT
touch .editorconfig
touch pyproject.toml
touch .isort.cfg
touch setup.cfg
touch tox.ini
