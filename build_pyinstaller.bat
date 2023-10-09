pipenv clean
pipenv sync

pipenv run pyinstaller dobble_maker_gui.py ^
    --clean ^
    --onefile ^
    --noconsole ^
    --name=dobble_maker_gui.exe ^
    --add-data ".venv/Lib/site-packages/galois/_databases/conway_polys.db;./galois/_databases" ^
    --add-data ".venv/Lib/site-packages/galois/_databases/irreducible_polys.db;./galois/_databases" ^
    --add-data ".venv/Lib/site-packages/galois/_databases/prime_factors.db;./galois/_databases" ^
