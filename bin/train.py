#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import modeling.parser
from modeling.utils import (
        build_model_id,
        build_model_path,
        setup_model_dir,
        load_model_json,
        setup_logging, 
        callable_print,
        save_model_info,
        ModelConfig)

def main(args):
    model_id = build_model_id(args)

    model_path = build_model_path(args, model_id)

    json_cfg = load_model_json(args)

    setup_model_dir(args, model_path)

    if 'persistent' in json_cfg['mode']:
        print('model path is %s' % model_path)

    if 'background' in json_cfg['mode']:
        callback_logger = logging.info
        sys.stdout, sys.stderr = setup_logging(
            os.path.join(model_path, 'model.log'))
        verbose = 0
    else:
        callback_logger = callable_print
        verbose = 1

    json_cfg['mode'] = args.mode
    json_cfg['model_path'] = model_path
    json_cfg['stdout'] = sys.stdout
    json_cfg['stderr'] = sys.stderr
    json_cfg['logger'] = callback_logger
    json_cfg['verbose'] = verbose

    config = ModelConfig(**json_cfg)

    if 'persistent' in config.mode:
        save_model_info(config, model_path)

    sys.path.append(args.model_dir)
    sys.path.append('.')
    import model
    from model import fit

    model.fit(config)

if __name__ == '__main__':
    parser = modeling.parser.build_keras()
    sys.exit(main(parser.parse_args()))
