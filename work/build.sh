
cd ..
WOCO_PATH=`pwd`
BUILD_PATH=$WOCO_PATH/build

mkdir $BUILD_PATH

cd $BUILD_PATH
cmake -DWITH_PYTHON=1 $WOCO_PATH
make -j

cp -p $BUILD_PATH/lib/*.so $WOCO_PATH/work/
cp -p $BUILD_PATH/bin/* $WOCO_PATH/work/
cp -p $BUILD_PATH/python/WOCO.py $WOCO_PATH/work/WOCO.py
