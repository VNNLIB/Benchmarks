@echo off
:: README:
::      This script assume that 7z is installed and added to the PATH, if not, you can add the path to 7z.exe modifying the pathTo7z variable:
:: set "pathTo7z=C:\Program Files\7-Zip\7z.exe"
set "pathTo7z=7z"
:: Set the parentDir to the directory where the batch file is located
set "parentDir=%~dp0"
set "parentDir=%parentDir:~0,-1%"

:: Recursively navigate through all directories, extract .gz files and delete the original .gz files
for /r "%parentDir%" %%F in (*.gz) do (
    echo Unzipping file: %%F
    "%pathTo7z%" x "%%F" -o"%%~dpF" -y
    del "%%F"
)

echo Done