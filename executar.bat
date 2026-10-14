@echo off
rem ============================================================
rem  PIVC — Detecção de Desperdício de Materiais em Obras
rem
rem  DOIS CLIQUES NESTE ARQUIVO = abre o front-end swipe no
rem  navegador (tema preto/amarelo). Mantenha a janela preta
rem  aberta enquanto estiver usando — ela é o servidor.
rem
rem  Comandos avançados (opcionais):
rem    executar.bat analisar "foto.jpg"   analisa e pede confirmação no terminal
rem    executar.bat painel                interface gráfica de janela (Tkinter)
rem    executar.bat status                estado da memória e do modelo
rem    executar.bat site --porta 8080     troca a porta do site
rem ============================================================
setlocal
title PIVC — servidor web (mantenha esta janela aberta)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Primeira execucao: criando ambiente virtual e instalando dependencias...
    echo Isso pode levar alguns minutos. Aguarde.
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install opencv-python numpy scikit-learn pillow flask
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" main.py site
) else (
    ".venv\Scripts\python.exe" main.py %*
)

echo.
echo Servidor encerrado. Pressione uma tecla para fechar esta janela.
pause >nul
endlocal
