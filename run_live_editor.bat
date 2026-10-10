@echo off

if exist "%~1\" (
    echo Running live eng editor...
) else (
    echo Please provide a valid directory. e.g. run_live_editor.bat path-to-image-directory
    exit /b 0
)

python -m manga_translator local --config-file ./editor-config.json --font-path fonts/anime_ace.ttf --overwrite --use-gpu -i "%~1"
