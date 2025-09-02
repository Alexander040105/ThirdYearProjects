c:
cd C:\Users\%USERNAME%\Desktop\
pause
mkdir GameProject
cd GameProject
mkdir Assets
mkdir Source
mkdir Docs

cd Assets
mkdir Images
mkdir Audio

cd..
cd Docs
echo This is the game project documentation > README.TXT

cd ..
cd Source
echo //Main game source file > main.cpp

cd ..
copy Docs\README.TXT Assets\

cd Source
move main.cpp ..

cd..
findstr /s "documentation" C:\GameProject\*.txt 

pause
