# UtilityBelt
For BAT tools. In order for the package to build properly, you must first install Proj 8.0.0 manually by following the 
directions below.


#### Step 1: Install GEOS
```
# Individual installation, just installing libgeos-dev might work but same result
apt-get install libgeos-dev libgeos++-dev libgeos-3.8.0 libgeos-c1v5 libgeos-doc
```
#### Step 2: Install proj dependencies
```
# Dependencies
apt install cmake
apt install sqlite3
apt install curl && apt-get install libcurl4-openssl-dev
```
#### Step 3: Install proj
```
# Using apt-get install proj-bin unfortunately only installs up to 6.3.1 as of writing, 
# so we build from source

# Navigate to the folder you'd like to install to, then run below
wget https://download.osgeo.org/proj/proj-8.0.0.tar.gz 
tar -xf proj-8.0.0.tar.gz
cd proj-8.0.0

# Build
mkdir build && cd build
cmake ..
cmake --build .
cmake --build . --target install

# Make sure it all installed right
ctest

# Move binaries
cp ./bin/* /bin
cp ./lib/* /lib

# If you're using a conda env, you may also need to copy to the conda bin and lib folders
cp ./bin/* ~/.conda/envs/{env}/bin
cp ./lib/* ~/.conda/envs/{env}/lib
```
#### Step 4: Install requirements
Once this is all completed, you should be able to successfully run
```
pip install -r requirements.txt
```
to build the environment. 