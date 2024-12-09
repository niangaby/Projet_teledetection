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
mc rm /home/onyxia/work/sigmam2_telea/init_env_telea_project.sh
mv /home/onyxia/work/sigmam2_telea /home/onyxia/work/libsigma


# Install OTB
#mkdir /home/onyxia/work/otb
#curl https://www.orfeo-toolbox.org/packages/archives/OTB/OTB-9.0.0-Linux.tar.gz -o /home/onyxia/work/otb/OTB-9.0.0-Linux.tar.gz
#tar xf /home/onyxia/work/otb/OTB-9.0.0-Linux.tar.gz --one-top-level=/home/onyxia/work/otb/OTB_install
#source /home/onyxia/work/otb/OTB_install/otbenv.profile

# Copy distant files
mc cp -r s3/mlang/diffusion/project data

# Install additional packages 
conda install rasterstats
conda install gdal
