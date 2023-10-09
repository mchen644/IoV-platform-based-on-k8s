apt-get update -y
apt-get install  cmake gcc g++ git -y
 
apt-get install libxkbcommon-dev -y
apt-get install wayland-protocols -y
apt install libglew-dev -y
cd Pangolin
rm -rf build
mkdir build
cd build
cmake ..
make -j30
make install
cd ../..

