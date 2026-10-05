@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m rbr2beamng.gui
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m rbr2beamng.gui
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo Python 3 was not found. Install Python or create the .venv environment first.
            pause
            exit /b 1
        )
        python -m rbr2beamng.gui
    )
)

if errorlevel 1 (
    echo.
    echo RBR2BeamNG failed to start. Review the error above.
    echo For missing-package errors, install uv and run: uv sync --locked
    pause
    exit /b 1
)
