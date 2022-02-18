#!/bin/bash
PYTHON_EXEC=${1}
source /users/micheleb/.bashrc
conda activate poetryGeneration
which python
echo "Executing python $PYTHON_EXEC"
python $PYTHON_EXEC
