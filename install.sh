# Install neccessary libraries
pip3 install -r requirements.txt

# Install gym-minigrid
cd envs
cd gym-minigrid
pip3 install --upgrade pip
pip3 install matplotlib cython
pip3 install -e .

# Install NeoRL envs
cd ../neorl-benchmark/
pip3 install -e .

# Install datasets
cd ..
cd ..
cd datasets
pip3 install -e .