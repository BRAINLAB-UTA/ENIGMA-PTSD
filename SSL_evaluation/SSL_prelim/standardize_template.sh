LINE=88

source ~/miniconda3/etc/profile.d/conda.sh

conda activate JMM_env

cd /home/mayortorresjm/ENIGMA-PTSD/Code/SSL_evaluation/

autoflake --in-place --recursive \
  --remove-all-unused-imports \
  --remove-unused-variables \
  --ignore-init-module-imports \
  --exclude .venv,venv,build,dist,.git,__pycache__ \
  *.py

autopep8 --in-place --recursive \
  --max-line-length "$LINE" \
  --aggressive --aggressive \
  --exclude .venv,venv,build,dist,.git,__pycache__ \
  *.py

ruff check *.py --fix
ruff format *.py

pylint *.py

##  be sure to always pass the code evaluation with 7-8 at least given by pylint at the end of the standard evaluation.
