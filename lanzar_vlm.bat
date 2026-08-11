@echo off
:: Launcher para el etiquetado VLM (Gemini)
:: Llama directamente al python.exe del venv

SET PYTHON="C:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\venv\Scripts\python.exe"
SET SCRIPT="C:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\data_pipeline\4_vlm_extractor.py"
SET LOG="C:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\logs\vlm_log.txt"

echo [%DATE% %TIME%] Iniciando tanda VLM... >> %LOG%
%PYTHON% %SCRIPT% >> %LOG% 2>&1
echo [%DATE% %TIME%] Tanda VLM finalizada con codigo: %ERRORLEVEL% >> %LOG%
