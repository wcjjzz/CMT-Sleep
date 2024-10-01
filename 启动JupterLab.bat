@echo off
call .\env\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    echo Activate failed
    pause
    exit /b
)
jupyter lab -p 8000
IF %ERRORLEVEL% NEQ 0 (
    echo Jupyter lab failed to start
    pause
    exit /b
)
pause
