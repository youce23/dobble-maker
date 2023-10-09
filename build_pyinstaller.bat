pipenv clean
pipenv sync

pipenv run pyinstaller dobble_maker_gui.py ^
    --clean ^
    --onefile ^
    --noconsole ^
    --name=dobble_maker_gui.exe ^
