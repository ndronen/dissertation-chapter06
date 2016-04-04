#!/bin/bash -e

train_cfg="n_epoch=100 patience=10 samples_per_epoch=10000"

for optimizer in Adam Adagrad SGD
do
    optimizer_cfg="optimizer=$optimizer"
    if [ $optimizer == "SGD" ]
    then
        optimizer_cfg="$optimzer_cfg learning_rate=0.1 decay=0.0 momentum=0.9"
    fi

    for gaussian_noise_sd in 0.0 0.01 0.03 0.1 0.3 1.0
    do
        noise_cfg="gaussian_noise_sd=$gaussian_noise_sd"

        for max_edits in 1 2
        do
            edit_cfg="min_edits=1 max_edits=$max_edits"
            model_cfg="$train_cfg $optimizer_cfg $noise_cfg $edit_cfg"
            echo bin/train.py models/binary --mode persistent-background --model-cfg "$model_cfg"
        done
    done
done
