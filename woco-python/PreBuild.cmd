%~d0
cd "%~dp0"
swig -MM woco.i
swig -c++ -python woco.i
