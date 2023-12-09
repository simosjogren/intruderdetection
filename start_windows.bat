@echo off

set VENV_DIR=./mlenv

if exist %VENV_DIR% (
    echo Activating existing virtual environment...
    call %VENV_DIR%/Scripts/activate
) else (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
    call %VENV_DIR%/Scripts/activate
    echo Installing requirements...
    pip install -r requirements.txt
)

echo Running main.py...
python main.py

echo Closing intruder detection