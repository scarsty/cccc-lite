cd ..
mkdir build
cd build
cmake ..
make -j
cd ../extension
cp -p ../build/lib/*.so .
cmake .
make
cp -p *.so ../work/
cp -p ../build/bin/* ../work/

