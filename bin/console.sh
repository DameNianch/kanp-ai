docker run -it --rm --gpus all \
    -v ${PWD}/python:/workdir/python -v ${PWD}/mldata:/workdir/mldata \
    kanp-ai \
    /bin/bash
