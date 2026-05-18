@echo off
setlocal

cd /d "%~dp0"

if not exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    echo No se encontro Miniconda en "%USERPROFILE%\miniconda3".
    echo Ajusta la ruta de activate.bat dentro de run_app.bat si tu Conda esta en otra carpeta.
    pause
    exit /b 1
)

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" rag-books
if errorlevel 1 (
    echo No se pudo activar el ambiente Conda rag-books.
    echo Crea el ambiente con: conda env create -f environment.yml
    pause
    exit /b 1
)

streamlit run app.py --server.port 8501 --server.headless false

pause
