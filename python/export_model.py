#!/usr/bin/python3
import sys
import os
import argparse
import random
import time
import logging
import json
import datetime
import struct

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np

from model import Model, ModelUtils
import common

#Command and args-------------------------------------------------------------------

description = """
Export neural net weights and graph to file.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)

args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]

loglines = []
def log(s):
  loglines.append(s)
  print(s,flush=True)

log("model_config_json" + ": " + str(model_config_json))
log("name_scope" + ": " + str(name_scope))


# Model ----------------------------------------------------------------
print("Building model", flush=True)
with open(model_config_json) as f:
  model_config = json.load(f)

pos_len = 19 # shouldn't matter, all we're doing is exporting weights that don't depend on this

with tf.compat.v1.variable_scope(name_scope):
  model = Model(model_config,pos_len,{})

  ModelUtils.print_trainable_variables(log)

  # Testing ------------------------------------------------------------

  print("Testing", flush=True)

  saver = tf.compat.v1.train.Saver(
    max_to_keep = 10000,
    save_relative_paths = True,
  )


  tfconfig = tf.compat.v1.ConfigProto(log_device_placement=False)

  with tf.compat.v1.Session(config=tfconfig) as session:
    saver.restore(session, model_variables_prefix)

    sys.stdout.flush()
    sys.stderr.flush()

    log("Began session")

    sys.stdout.flush()
    sys.stderr.flush()

    bin_inputs = model.bin_inputs
    global_inputs = model.global_inputs
    symmetries = model.symmetries
    include_histroy = model.include_history


    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = ["swa_model/policy_output","swa_model/value_output"]

    output_graph_def = graph_util.convert_variables_to_constants(session, input_graph_def, output_node_names)
    # For some models, we would like to remove training nodes
    output_graph_def = graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

    with tf.gfile.GFile('frozen.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())