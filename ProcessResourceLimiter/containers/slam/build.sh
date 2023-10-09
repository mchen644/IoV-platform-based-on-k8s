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

apt-get install libeigen3-dev -y
apt-get install mlocate -y
apt-get install build-essential -y
apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev -y

cd opencv3
rm -rf build
mkdir build
cd build
cmake ..
make -j30
make install
cd ../..

echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf

ldconfig

PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH

source /etc/bash.bashrc
updatedb


cd ORB_SLAM2
chmod +x build.sh
./build.sh
