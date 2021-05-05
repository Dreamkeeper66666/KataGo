#!/bin/bash -eu
set -o pipefail
{
#Takes any models in tfsavedmodels_toexport/ and outputs a cuda-runnable model file to modelstobetested/
#Takes any models in tfsavedmodels_toexport_extra/ and outputs a cuda-runnable model file to models_extra/
#Should be run periodically.

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR USEGATING"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as export_model.py."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper and output directly to models/"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIR="$1"
shift
MODELKIND="$1"
shift

GITROOTDIR="$(git rev-parse --show-toplevel)"


#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/tfsavedmodels_toexport
mkdir -p "$BASEDIR"/tfsavedmodels_toexport_extra
mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/models_extra
mkdir -p "$BASEDIR"/models

function exportStuff() {
    FROMDIR="$1"
    TODIR="$2"

    set -x
    python3 ./export_model.py \
            --checkpoint "$FROMDIR"/checkpoints/*.ckpt \
            --export-dir "$FROMDIR"/checkpoints \
            --model-name "$NAMEPREFIX""-latest" \
            --format "bin" \
            --filename-prefix "$NAMEPREFIX""-latest" \
            --model-config "$GITROOTDIR"/python/torch/modelconfigs/v8_"$MODELKIND".json  
    set +x


    gzip "$FROMDIR"/checkpoints/"$NAMEPREFIX""-""latest".bin

    #Sleep a little to allow some tolerance on the filesystem
    sleep 2

    echo "Done exporting:" "$NAMEPREFIX""-latest"  "to" "$BASEDIR""/models"
    
    mv "$FROMDIR"/checkpoints/"$NAMEPREFIX""-""latest".bin.gz "$BASEDIR"/models
    

}




exportStuff "$BASEDIR"/train/"$MODELKIND"/lightning_logs/version_18 "$BASEDIR"/models


exit 0
}
