#!/bin/bash -e

train_cfg="n_epoch=50 patience=10 samples_per_epoch=10000 optimizer=Adam"

for editor in LearnedEditTransformer RandomWordTransformer
do
    editor_cfg="non_word_generator=$editor"
    for max_edits in 1 2 3
    do
        edit_cfg="min_edits=1 max_edits=$max_edits"
        model_dest="models/binary/convolutional_context_noise/"$(echo $editor_cfg $edit_cfg | sed 's, ,_,g')
        model_cfg="$train_cfg $optimizer_cfg $editor_cfg $edit_cfg"
        echo bin/train.py models/binary --mode persistent-background --model-dest $model_dest --model-cfg $model_cfg
    done
done | parallel --gnu --jobs 4 --colsep '\t' --verbose {1}
