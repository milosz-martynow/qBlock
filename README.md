# Dependencies

## Operational system
Operating System in which `qBlock` developed is Windows 11 Pro. For this environment this `README.md` is written.
Nevertheless, it should be not a problem to follow below command with small changes to run it under Linux OS.

## Python 
Python in version [3.12.7](https://peps.python.org/pep-0693/). 

For better maintenance of Python code it is worth to use 
[Python Virtual Environment](https://docs.python.org/3/library/venv.html).

You can create Python Virtual Environment by typing in terminal of your project root folder:
```
py.exe -m venv .venv
.venv\Scripts\activate
```
Note - `'.venv'` name is included in `.gitignore` file.

### PIP
Upgrading pip will be useful, when issues with requirements libraries araises:
```
python.exe -m pip install --upgrade pip
```

## Install packages
To install `qBlock` with all project dependencies:
```
pip.exe install -e .
```

Sometimes, python does not come with `setuptools`. If so - above will not work until
`setuptools` will be installed virtual environment:
```
pip.exe install setuptools==80.9.0
```
## Test and formatting:
To test and format code type in terminal:
```
isort.exe .
black.exe --config=.blackrc .\q_block\ .\tests\ setup.py
pylint.exe --rcfile=.pylintrc .\q_block\ .\tests\ setup.py
pytest.exe .
```
