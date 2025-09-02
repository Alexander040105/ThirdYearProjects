cd C:\Users\%USERNAME%\Desktop\
pause

if not exist Test_Files (
    mkdir Test_Files
)

cd Test_Files
setlocal enabledelayedexpansion

set counter=1
for %%E in (txt log csv html xml json bat exe pdf docx) do (
	for /l %%N in (1,1,10) do (
		echo This is the test file %%N of type %%E > test!counter!.%%E
		set /a counter+=1
	)
)

echo 100 test files created successfully!
pause