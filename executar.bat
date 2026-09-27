@echo off
rem ============================================================
rem  PIVC - Detecção de Desperdício de Materiais em Obras
rem  Executa o sistema sempre com o Python do ambiente virtual
rem  do projeto (.venv), ignorando os outros Pythons do PC.
rem
rem  Uso:   executar.bat <comando> [opções]
rem  Ex.:   executar.bat analisar "foto.jpg"
rem         executar.bat status
rem  Sem comando = abre a interface gráfica (painel).
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Primeira execucao: criando ambiente virtual e instalando dependencias...
    echo Isso pode levar alguns minutos. Aguarde.
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install opencv-python numpy scikit-learn pillow joblib
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" main.py painel
) else (
    ".venv\Scripts\python.exe" main.py %*
)
endlocal
