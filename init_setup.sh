#for linux sysetm users

echo [${date}]: "START"

echo [$(date)]: --"creating env with python 3.8 version"

conda create --prefix ./ppenv python=3.8 -y

echo [$(date)]:"activating the environment"

source activate ./ppenv

echo [$(date)]:"installing the dev requirements"

pip install -r requirements.txt

echo [$(date)]: "END"
