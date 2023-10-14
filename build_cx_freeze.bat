pipenv clean
pipenv sync --dev

pipenv run python .\setup_cx_freeze.py build

rmdir /s /q release
mkdir release
git checkout release

xcopy /y /i LICENSE release
xcopy /y /s /i .\build\exe.win-amd64-3.11\* release
xcopy /y /i samples release\samples
