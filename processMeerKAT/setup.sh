dir=$(dirname $BASH_SOURCE)
export PATH=$dir:$PATH
export PYTHONPATH=$dir:$PYTHONPATH
export SINGULARITYENV_PYTHONPATH="$dir:\$PYTHONPATH"
git config --global --add safe.directory $dir
