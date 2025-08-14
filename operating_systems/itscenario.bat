c:
cd C:\Users\%USERNAME%\Desktop\
pause
if not exist Messy_Files (
    mkdir Messy_Files
)

cd Messy_Files

if not exist Documents (
    mkdir Documents
)
if not exist Images (
    mkdir Images
)
if not exist Scripts (
    mkdir Scripts
)

set "docCount=0"
set "imgCount=0"
set "scriptCount=0"
for /l %%i in (1,1,15) do echo This is text file number %%i > "text_file%%i.txt"
for /l %%i in (1,1,10) do type nul > "report%%i.docx"
for /l %%i in (1,1,10) do type nul > "image%%i.jpg"
for /l %%i in (1,1,5) do type nul > "photo%%i.png"
for /l %%i in (1,1,10) do echo @echo off > "script%%i.bat"

for %%f in ("C:\Users\%USERNAME%\Desktop\Messy_Files\*.*") do (
    if /i "%%~xf"==".txt" (
        move "%%f" "C:\Users\%USERNAME%\Desktop\Messy_Files\Documents"
        set /a "docCount+=1"
    ) else if /i "%%~xf"==".docx" (
        move "%%f" "C:\Users\%USERNAME%\Desktop\Messy_Files\Documents"
        set /a "docCount+=1"
    ) else if /i "%%~xf"==".jpg" (
        move "%%f" "C:\Users\%USERNAME%\Desktop\Messy_Files\Images"
        set /a "imgCount+=1"
    ) else if /i "%%~xf"==".png" (
        move "%%f" "C:\Users\%USERNAME%\Desktop\Messy_Files\Images"
        set /a "imgCount+=1"
    ) else if /i "%%~xf"==".bat" (
        move "%%f" "C:\Users\%USERNAME%\Desktop\Messy_Files\Scripts"
        set /a "scriptCount+=1"
    )
)
pause

echo Leftover files in Messy_Files:
dir /b "C:\Users\%USERNAME%\Desktop\Messy_Files"

rem Create cleanup_log.txt on Desktop with current date/time
(
    echo %date% %time%
    echo Documents folder contents:
    dir /b "C:\Users\%USERNAME%\Desktop\Messy_Files\Documents"
    echo Total Documents: %docCount%
) > "C:\Users\%USERNAME%\Desktop\Messy_Files\cleanup_log.txt"

echo Cleanup complete. Log saved to Desktop\cleanup_log.txt
pause