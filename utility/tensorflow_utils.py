# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import random
import numpy as np
import tensorflow as tf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_model_vars(scope = None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        
        # Error (EfficientNet) : <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
        # print('var:', var, type(var))
        # print('ema_var:', ema_var, type(ema_var))

        if ema_var is not None:
            return ema_var
        else:
            return var

        return ema_var if ema_var else var

    return ema_getter

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name = 'prefix')

    return graph

def calculate_FLOPs(graph):
    flops = tf.profiler.profile(graph, options = tf.profiler.ProfileOptionBuilder.float_operation())
    total_flops = flops.total_float_ops

    # MFLOPS 10**6
    # GFLOPS 10**9
    mflops = total_flops / 10**6
    
    if mflops > 1000:
        gflops = mflops / 1000
        flops_str = '{:.1f}G'.format(mflops)
    else:
        flops_str = '{:.0f}M'.format(mflops)

    return total_flops, flops_str

# file_path = './wider_resnet_28_large.txt'
def model_summary(vars, graph, file_path = None): 
    def shape_parameters(shape):
        v = 1
        for s in shape:
            v *= s
        return v
    
    with open(file_path, 'w') as f:
        f.write('_' * 100 + '\n')
        f.write('{:50s} {:20s} {:20s}'.format('Name', 'Shape', 'Param #') + '\n')
        f.write('_' * 100 + '\n')

        model_params = 0
        
        for var in vars:
            shape = var.shape.as_list()
            params = shape_parameters(shape)

            model_params += params

            f.write('{:50s} {:20s} {:20s}'.format(var.name, str(shape), str(params)) + '\n')
            f.write('_' * 100 + '\n')

        million = model_params / 1000000
        if million >= 1:
            million = str(int(million))
        else:
            million = '{:2f}'.format(million)

        total_flops, flops_str = calculate_FLOPs(graph)
        
        f.write('Total Params : {:,}, {}M'.format(model_params, million) + '\n')
        f.write('Total FLOPs : {:,}, {}'.format(total_flops, flops_str) + '\n')
        f.write('_' * 100 + '\n')

def KL_Divergence_with_logits(p_logits, q_logits):
    p = tf.nn.softmax(p_logits)

    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    kl = tf.reduce_sum(p * (log_p - log_q), axis = -1)
    return kl

def interleave(x, batch):
    s = x.get_shape().as_list()
    return tf.reshape(tf.transpose(tf.reshape(x, [-1, batch] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] + s[1:])

def de_interleave(x, batch):
    s = x.get_shape().as_list()
    return tf.reshape(tf.transpose(tf.reshape(x, [batch, -1] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] +s[1:])

if __name__ == '__main__':
    p_logits = [
        [0.5, 0.1515, 12],
        [0.5, 0.1515, 12]
    ]
    q_logits = [
        [0.5, 0.1515, 12],
        [0.5, 10, 14]
    ]
    
    loss = KL_Divergence_with_logits(p_logits, q_logits)
    print(loss)

    sess = tf.Session()
    print(sess.run(loss))

