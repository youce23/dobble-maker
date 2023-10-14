pipenv clean
pipenv sync --dev

pipenv run create-version-file version.yaml --outfile app.version

pipenv run pyinstaller ^
    --clean ^
    --onefile ^
    --noconsole ^
    --name dobble_maker_gui.exe ^
    --version-file app.version ^
    --exclude-module altgraph ^
    --exclude-module colorama ^
    --exclude-module contourpy ^
    --exclude-module deprecation ^
    --exclude-module fonttools ^
    --exclude-module jinja2 ^
    --exclude-module markupsafe ^
    --exclude-module lxml ^
    --exclude-module pefile ^
    --exclude-module pikepdf ^
    --exclude-module pillow ^
    --exclude-module pip ^
    --exclude-module pyinstaller ^
    --exclude-module pyinstaller-hooks-contrib ^
    --exclude-module pyinstaller-versionfile ^
    --exclude-module python-dateutil ^
    --exclude-module pywin32-ctypes ^
    --exclude-module setuptools ^
    --exclude-module typing-extensions ^
    --exclude-module wheel ^
    --add-data ".venv/Lib/site-packages/galois/_databases/conway_polys.db;./galois/_databases" ^
    --add-data ".venv/Lib/site-packages/galois/_databases/irreducible_polys.db;./galois/_databases" ^
    --add-data ".venv/Lib/site-packages/galois/_databases/prime_factors.db;./galois/_databases" ^
    dobble_maker_gui.py

move /y .\dist\dobble_maker_gui.exe .\release
copy /y LICENSE .\release\LICENSES
xcopy /y /i samples release\samples
