dir=$(dirname $BASH_SOURCE)/processMeerKAT
export PATH=$dir:$PATH
export PYTHONPATH=$dir:$PYTHONPATH
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH:\$PYTHONPATH"
git config --global --add safe.directory $(dirname $BASH_SOURCE)
