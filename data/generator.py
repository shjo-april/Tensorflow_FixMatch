# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import sys

import threading
import tensorflow as tf

from utility.timer import *

class Generator(threading.Thread):
    def __init__(self, option):
        super().__init__()
        self.daemon = True
        
        self.labeled_batch_loader = option['labeled_batch_loader']
        self.unlabeled_batch_loader = option['unlabeled_batch_loader']

        self.placeholders = option['placeholders']
        
        self.queue = tf.FIFOQueue(
            capacity = option['queue_size'],
            dtypes = [ph.dtype for ph in self.placeholders],
            shapes = [ph.get_shape().as_list() for ph in self.placeholders],
        )
        
        self.enqueue_op = self.queue.enqueue(self.placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues = True)
        
        self.sess = None
        self.coord = None
    
    def set_session(self, sess):
        self.sess = sess
    
    def set_coordinator(self, coord):
        self.coord = coord

    def size(self):
        return self.sess.run(self.queue.size())

    def run(self):
        with self.sess.as_default():
            try:
                while not self.coord.should_stop():
                    try:
                        self.timer = Timer()
                        self.timer.tik()
                        
                        while True:
                            data = self.labeled_batch_loader.main_queue.get()
                            data += self.unlabeled_batch_loader.main_queue.get()

                            self.enqueue_op.run(feed_dict = dict(zip(self.placeholders, data)))
                            
                            del data
                    
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                        print('[!] coord exception')
                        sys.exit(-1)
                    
                    except Exception as e:
                        print('[!] Exception = {}'.format(str(e)))
                        sys.exit(-1)
            
            except Exception as e:
                print('[!] Exception = {}'.format(str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
    
    def dequeue(self):
        return self.queue.dequeue()