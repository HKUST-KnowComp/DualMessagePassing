import pathlib
import os
import re

INF = 1e30
_INF = -1e30
EPS = 1e-8
PI = 3.141592653589793

LEAKY_RELU_A = 1 / 5.5

LOOPFLAG = "is_loop"
REVFLAG = "is_reversed"
NORM = "norm"
INDEGREE = "in_deg"
INNORM = "in_norm"
OUTDEGREE = "out_deg"
OUTNORM = "out_norm"
NODEID = "id"
EDGEID = "id"
NODELABEL = "label"
EDGELABEL = "label"
NODEEIGENV = "node_eigenv"
EDGEEIGENV = "edge_eigenv"
NODEFEAT = "node_feat"
EDGEFEAT = "edge_feat"
NODETYPE = "node_type"
EDGETYPE = "edge_type"
NODEMSG = "node_msg"
EDGEMSG = "edge_msg"
NODEAGG = "node_agg"
EDGEAGG = "edge_agg"
NODEOUTPUT = "node_out"
EDGEOUTPUT = "edge_out"

INIT_STEPS = 600
SCHEDULE_STEPS = 10000
NUM_CYCLES = 2
MIN_PERCENT = 1e-3
