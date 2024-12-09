#!/bin/bash

# This init script clones a Git repository that contains a Jupyter notebook
# named `tutorial.ipynb` and opens it in Jupyter Lab at startup
# Expected parameters : None

# Clone repository and give permissions to the onyxia user
GIT_REPO=sigmam2_telea
git clone --depth 1 https://framagit.org/cours1/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/
mc rm /home/onyxia/work/sigmam2_telea/requirement.txt
mc rm /home/onyxia/work/sigmam2_telea/init_env_telea.sh
mv /home/onyxia/work/sigmam2_telea /home/onyxia/work/libsigma


# Install OTB
mkdir /home/onyxia/work/otb
curl https://www.orfeo-toolbox.org/packages/archives/OTB/OTB-9.0.0-Linux.tar.gz -o /home/onyxia/work/otb/OTB-9.0.0-Linux.tar.gz
tar xf /home/onyxia/work/otb/OTB-9.0.0-Linux.tar.gz --one-top-level=/home/onyxia/work/otb/OTB_install
source /home/onyxia/work/otb/OTB_install/otbenv.profile
sed -i '51i OTB_INSTALL_DIR="/home/onyxia/work/otb/OTB_install"' /home/onyxia/work/otb/OTB_install/otbenv.profile
echo "source /home/onyxia/work/otb/OTB_install/otbenv.profile" >> /home/onyxia/.bashrc
#cat /home/onyxia/work/otb/OTB_install/otbenv.profile >> /home/onyxia/.bashrc

# Copy distant files

#echo "c.LabApp.default_url = 'sigmam2_telea/My_first.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py
mc mv /home/onyxia/work/sigmam2_telea/My_first.ipynb /home/onyxia/work/My_first.ipynb
echo "c.LabApp.default_url = '/lab/tree/My_first.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py

# Install additional packages 


