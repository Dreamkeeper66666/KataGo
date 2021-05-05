#!/bin/bash -eu
set -o pipefail
{
#Runs tensorflow training in $BASEDIR/train/$TRAININGNAME
#Should be run once per persistent training process.
#Outputs results in tfsavedmodels_toexport/ in an ongoing basis (EXPORTMODE == "main").
#Or, to tfsavedmodels_toexport_extra/ (EXPORTMODE == "extra").
#Or just trains without exporting (EXPORTMODE == "trainonly").

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODELKIND BATCHSIZE EXPORTMODE OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like b10c128, see ../modelconfigs.py"
    echo "BATCHSIZE number of samples to concat together per batch for training, must match shuffle"
    echo "EXPORTMODE 'main': train and export for selfplay. 'extra': train and export extra non-selfplay model. 'trainonly': train without export"
    exit 0
fi
BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
BATCHSIZE="$1"
shift
EXPORTMODE="$1"
shift

GITROOTDIR="$(git rev-parse --show-toplevel)"

#------------------------------------------------------------------------------
set -x

mkdir -p "$BASEDIR"/train/"$TRAININGNAME"
git show --no-patch --no-color > "$BASEDIR"/train/"$TRAININGNAME"/version.txt
git diff --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diff.txt
git diff --staged --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diffstaged.txt

# For archival and logging purposes - you can look back and see exactly the python code on a particular date
DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$BASEDIR"/scripts/train/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"
cp "$GITROOTDIR"/python/torch/*.py "$GITROOTDIR"/python/torch/selfplay/train.sh "$DATED_ARCHIVE"
git show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt
git diff --no-color > "$DATED_ARCHIVE"/diff.txt
git diff --staged --no-color > "$DATED_ARCHIVE"/diffstaged.txt


if [ "$EXPORTMODE" == "main" ]
then
    EXPORT_SUBDIR=tfsavedmodels_toexport
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "extra" ]
then
    EXPORT_SUBDIR=tfsavedmodels_toexport_extra
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "trainonly" ]
then
    EXPORT_SUBDIR=tfsavedmodels_toexport_extra
    EXTRAFLAG="-no-export"
else
    echo "EXPORTMODE was not 'main' or 'extra' or 'trainonly', run with no arguments for usage"
    exit 1
fi

time python3.7 "$GITROOTDIR"/python/torch/train.py \
     --run-name "$BASEDIR"/train/"$MODELKIND" \
     --data-dir "$BASEDIR"/shuffleddata/current/train \
     --model-config "$GITROOTDIR"/python/torch/modelconfigs/v8_"$MODELKIND".json  \
     --max_epochs 10000 \
     --gpus 1 \
     --resume ""$BASEDIR"/train/"$MODELKIND"/lightning_logs/version_1"/checkpoints/*.ckpt \
     $EXTRAFLAG \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/"$TRAININGNAME"/stdout.txt

exit 0
}
