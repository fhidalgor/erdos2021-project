# erdos2021-project
Final project for the Erdos Institute 2021 data-science bootcamp

# Installation

0.
 
> git clone --recursive git@github.com:fhidalgor/erdos2021-project.git
    
1. Install `poetry` and `pyenv`, you can get more information on how to do it on your local machine here:

* https://python-poetry.org/

* https://github.com/pyenv/pyenv

2. From your local repo folder install `python` with `pyenv`:

> pyenv install 3.8.3

> pyenv local 3.8.3

3. Configure `poetry` to use the right `python` version:

> poetry env use 3.8.3

> poetry install

4. Run compile to make sure everything is ok.
> ./compile.sh

5. Run tests. Make sure your variables are set in `env` file.
> ./test.sh

# Code style YAPF
To format the code, we use yapf. If you want to format a single file, type the following on the terminal:
> poetry run yapf -i path/to/file.py

To format an entire folder, use the following command:
> poetry run yapf -i -r path/to/folder/

# VS Code
To get the vs code debugger to work:
Create a file thats just named ".env" in engine-runner/ that has all the environment vars.
In the debugger click on the settings wheel and that should take you to a file called launch.json
put this in the configurations list
>        {
            "name": "Python: Current File w/ env",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        }
