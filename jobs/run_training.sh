PYTHON_EXEC=${1}
source /home/${USER}/.bashrc
conda activate finnishPoetryGeneration
echo "Executing $PYTHON_EXEC"
python $PYTHON_EXEC
