@echo off
REM Build help-doc HTML from Markdown sources using pandoc.
REM Run this after editing any help_docs\src\*.md file.
setlocal
set "HERE=%~dp0"
python "%HERE%build_help.py" html
if errorlevel 1 (
    echo [error] help-doc build failed
    exit /b 1
)
echo [ok] help-doc HTML rebuilt.
endlocal
