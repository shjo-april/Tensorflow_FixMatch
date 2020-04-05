# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

def flags_to_dict(flags):
    return {k : flags[k].value for k in flags}

def get_config():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ###############################################################################
    # Default Config
    ###############################################################################
    flags.DEFINE_string('experimenter', 'JSH', 'unknown')
    flags.DEFINE_string('use_gpu', '0', 'unknown')
    
    flags.DEFINE_integer('seed', 0, 'unknown')
    flags.DEFINE_integer('number_of_labels', 250, 'unknown')
    
    ###############################################################################
    # Dataset
    ###############################################################################
    flags.DEFINE_integer('number_of_loading_dataset', 10000, 'unknown')
    
    flags.DEFINE_integer('number_of_loader', 1, 'unknown')
    flags.DEFINE_integer('number_of_batch_loader', 2, 'unknown')
    
    flags.DEFINE_integer('max_size_of_loader', 2, 'unknown')
    flags.DEFINE_integer('max_size_of_batch_loader', 2, 'unknown')
    
    # labeled dataset
    flags.DEFINE_integer('number_of_labeled_decoder', 1, 'unknown')
    flags.DEFINE_integer('max_size_of_labeled_decoder', 16, 'unknown')

    # unlabeled dataset
    flags.DEFINE_integer('number_of_unlabeled_decoder', 32, 'unknown')
    flags.DEFINE_integer('max_size_of_unlabeled_decoder', 2048, 'unknown')
    
    ###############################################################################
    # Training Schedule
    ###############################################################################
    flags.DEFINE_float('init_learning_rate', 0.03, 'unknown')
    
    flags.DEFINE_integer('batch_size', 64, 'unknown')
    
    flags.DEFINE_integer('log_iteration', 100, 'unknown')
    flags.DEFINE_integer('valid_iteration', 10000, 'unknown')
    
    flags.DEFINE_integer('max_epochs', 1<<10, 'unknown')
    flags.DEFINE_integer('train_kimgs', 1<<16, 'unknown')
    
    flags.DEFINE_integer('max_iteration', 1024*1024, 'unknown')
    
    ###############################################################################
    # Training Technology for FixMatch
    ###############################################################################
    flags.DEFINE_integer('unlabeled_ratio', 7, 'unknown')

    flags.DEFINE_string('weak_augmentation', 'flip_and_crop', 'None/flip_and_crop/randaugment')
    flags.DEFINE_string('strong_augmentation', 'randaugment', 'None/flip_and_crop/randaugment')
    
    flags.DEFINE_float('ema_decay', 0.999, 'unknown')
    flags.DEFINE_float('weight_decay', 0.0005, 'unknown')

    flags.DEFINE_float('lambda_u', 1.0, 'unknown')
    flags.DEFINE_float('confidence_threshold', 0.95, 'unknown')

    ###############################################################################
    # Testing
    ###############################################################################
    flags.DEFINE_string('ckpt_path', None, 'unknown')

    return FLAGS

if __name__ == '__main__':
    import json
    
    flags = get_config()

    print(flags.use_gpu)
    print(flags_to_dict(flags))
    
    # print(flags.mixup)
    # print(flags.efficientnet_option)
