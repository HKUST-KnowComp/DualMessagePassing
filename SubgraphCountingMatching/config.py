import argparse

from utils.act import supported_act_funcs
from utils.io import str2bool, str2value, str2list, load_config, save_config
from utils.scheduler import supported_schedulers


def _add_cnn_config(parser):
    parser.add_argument(
        "--rep_cnn_batch_norm",
        type=str2bool,
        default=False,
        help="batch normalization in CNN"
    )

    parser.add_argument(
        "--rep_cnn_kernel_sizes",
        type=str2list,
        default=2,
        help="kernel sizes for convolutions and pooling"
    )

    parser.add_argument(
        "--rep_cnn_paddings",
        type=str2list,
        default=-1,
        help="paddings for convolutions and pooling"
    )

    parser.add_argument(
        "--rep_cnn_strides",
        type=str2list,
        default=1,
        help="strides for convolutions"
    )


def _add_rnn_config(parser):
    parser.add_argument(
        "--rep_rnn_layer_norm",
        type=str2bool,
        default=False,
        help="layer normalization in RNN"
    )

    parser.add_argument(
        "--rep_rnn_type",
        type=str,
        default="LSTM",
        help="RNN type (LSTM, GRU)"
    )

    parser.add_argument(
        "--rep_rnn_bidirectional",
        type=str2bool,
        default=False,
        help="bidirectional or unidirectional RNN"
    )


def _add_txl_config(parser):
    parser.add_argument(
        "--rep_txl_layer_norm",
        type=str2bool,
        default=True,
        help="layer normalization in TXL"
    )

    parser.add_argument(
        "--rep_txl_pre_norm",
        type=str2bool,
        default=True,
        help="pre-normalization in TXL"
    )

    parser.add_argument(
        "--rep_txl_num_heads",
        type=int,
        default=4,
        help="number of heads in attention"
    )

    parser.add_argument(
        "--rep_txl_seg_len",
        type=int,
        default=64,
        help="segment length in TXL"
    )

    parser.add_argument(
        "--rep_txl_mem_len",
        type=int,
        default=64,
        help="memory length in TXL"
    )

    parser.add_argument(
        "--rep_txl_clamp_len",
        type=int,
        default=-1,
        help="clamp length in TXL"
    )


def _add_rgcn_config(parser):
    parser.add_argument(
        "--rep_rgcn_batch_norm",
        type=str2bool,
        default=False,
        help="batch normalization in RGCN"
    )

    parser.add_argument(
        "--rep_rgcn_regularizer",
        type=str,
        default="bdd",
        choices=["none", "basis", "bdd", "diag", "scalar"],
        help="regularizer in relation decomposition in RGCN"
    )

    parser.add_argument(
        "--rep_rgcn_num_bases",
        type=int,
        default=4,
        help="number of bases in decomposition in RGCN"
    )

    parser.add_argument(
        "--rep_rgcn_edge_norm",
        type=str,
        default="in",
        choices=["none", "in", "out", "both"],
        help="edge norm in RGCN"
    )


def _add_rgin_config(parser):
    parser.add_argument(
        "--rep_rgin_batch_norm",
        type=str2bool,
        default=False,
        help="batch normalization in RGIN"
    )

    parser.add_argument(
        "--rep_rgin_regularizer",
        type=str,
        default="bdd",
        choices=["none", "basis", "bdd", "diag", "scalar"],
        help="regularizer in relation decomposition in RGIN"
    )

    parser.add_argument(
        "--rep_rgin_num_bases",
        type=int,
        default=4,
        help="number of bases in decomposition in RGIN"
    )

    parser.add_argument(
        "--rep_rgin_num_mlp_layers",
        type=int,
        default=2,
        help="number of MLP layers in RGIN"
    )


def _add_compgcn_config(parser):
    parser.add_argument(
        "--rep_compgcn_comp_opt",
        type=str,
        default="corr",
        choices=["sub", "mult", "corr"],
        help="composition operator in CompGCN"
    )

    parser.add_argument(
        "--rep_compgcn_edge_norm",
        type=str,
        default="none",
        choices=["none", "in", "out", "both"],
        help="edge norm in CompGCN"
    )

    parser.add_argument(
        "--rep_compgcn_batch_norm",
        type=str2bool,
        default=False,
        help="batch normalization in CompGCN"
    )


def _add_dmpnn_config(parser):
    parser.add_argument(
        "--rep_dmpnn_num_mlp_layers",
        type=int,
        default=2,
        help="number of MLP layers in dmpnn"
    )

    parser.add_argument(
        "--rep_dmpnn_batch_norm",
        type=str2bool,
        default=False,
        help="batch normalization in dmpnn"
    )


def _add_lrp_config(parser):
    parser.add_argument(
        "--lrp_seq_len",
        type=int,
        default=4,
        help="LRP truncated-BFS size"
    )

    parser.add_argument(
        "--rep_lrp_batch_norm",
        type=str2bool,
        default=False,
        help="batch normalization in LRP"
    )


def add_emb_net_config(parser):
    parser.add_argument(
        "--enc_base",
        type=int,
        default=2,
        help="base for encoding"
    )

    parser.add_argument(
        "--enc_net",
        type=str,
        default="Multihot",
        choices=["Multihot", "Position"],
        help="embedding network"
    )

    parser.add_argument(
        "--emb_net",
        type=str,
        default="Equivariant",
        choices=["Orthogonal", "Uniform", "Normal", "Equivariant"],
        help="embedding network"
    )

    parser.add_argument(
        "--filter_net",
        type=str,
        default="ScalarFilter",
        choices=["None", "ScalarFilter"],
        help="filter network"
    )

    parser.add_argument(
        "--share_emb_net",
        type=str2bool,
        default=True,
        help="whether to share embedding networks"
    )


def add_rep_net_config(parser):
    _add_cnn_config(parser)
    _add_rnn_config(parser)
    _add_txl_config(parser)
    _add_rgcn_config(parser)
    _add_rgin_config(parser)
    _add_compgcn_config(parser)
    _add_dmpnn_config(parser)
    _add_lrp_config(parser)

    parser.add_argument(
        "--rep_net",
        type=str,
        default="CNN",
        choices=["CNN", "RNN", "TXL", "RGCN", "RGIN", "CompGCN", "DMPNN", "LRP", "DMPLRP"],
        help="representation network"
    )

    parser.add_argument(
        "--rep_num_heads",
        type=int,
        default=4,
        help="number of heads for attention in RepNet"
    )

    parser.add_argument(
        "--rep_num_pattern_layers",
        type=int,
        default=3,
        help="number of pattern layers in RepNet"
    )

    parser.add_argument(
        "--rep_act_func",
        type=str,
        default="leaky_relu",
        choices=supported_act_funcs.keys(),
        help="activation function in RepNet"
    )

    parser.add_argument(
        "--rep_num_graph_layers",
        type=int,
        default=3,
        help="number of graph layers in RepNet"
    )

    parser.add_argument(
        "--rep_residual",
        type=str2bool,
        default=True,
        help="whether to use residual connections"
    )

    parser.add_argument(
        "--rep_dropout",
        type=float,
        default=0.0,
        help="dropout rate in RepNet"
    )

    parser.add_argument(
        "--share_rep_net",
        type=str2bool,
        default=True,
        help="whether to share representation networks"
    )


def add_pred_net_config(parser):
    parser.add_argument(
        "--pred_net",
        type=str,
        default="SumPredictNet",
        choices=[
            "SumPredictNet", "MeanPredictNet", "MaxPredictNet",
            "SumAttnPredictNet", "MeanAttnPredictNet", "MaxAttnPredictNet",
            "SumMemAttnPredictNet", "MeanMemAttnPredictNet", "MaxMemAttnPredictNet",
            "DIAMNet"
        ],
        help="representation network"
    )

    parser.add_argument(
        "--pred_with_enc",
        type=str2bool,
        default=True,
        help="whether to add encoding to predict"
    )

    parser.add_argument(
        "--pred_with_deg",
        type=str2bool,
        default=True,
        help="whether to add degree to predict"
    )

    parser.add_argument(
        "--pred_hid_dim",
        type=int,
        default=64,
        help="hidden dimension in PredNet"
    )

    parser.add_argument(
        "--pred_act_func",
        type=str,
        default="leaky_relu",
        choices=supported_act_funcs.keys(),
        help="activation function in PredNet"
    )

    parser.add_argument(
        "--pred_num_heads",
        type=int,
        default=4,
        help="number of heads for attention to predict"
    )

    parser.add_argument(
        "--pred_mem_len",
        type=int,
        default=4,
        help="numbre of memory blocks"
    )

    parser.add_argument(
        "--pred_mem_init",
        type=str,
        default="mean",
        choices=[
            "sum", "mean", "max",
            "attn", "lstm",
            "circular_sum", "circular_mean", "circular_max",
            "circular_attn", "circular_lstm"
        ],
        help="the way to initialize memory"
    )

    parser.add_argument(
        "--pred_infer_steps",
        type=int,
        default=3,
        help="the steps to inference in PredNet"
    )

    parser.add_argument(
        "--pred_dropout",
        type=float,
        default=0.0,
        help="dropout rate in PredNet"
    )


def add_model_config(parser):
    add_emb_net_config(parser)
    add_rep_net_config(parser)
    add_pred_net_config(parser)

    parser.add_argument(
        "--hid_dim",
        type=int,
        default=64,
        help="hidden dimension"
    )

    parser.add_argument(
        "--gnn_add_node_id",
        type=str2bool,
        default=False,
        help="whether to add node id information in node embedding"
    )

    parser.add_argument(
        "--gnn_add_edge_id",
        type=str2bool,
        default=False,
        help="whether to add source and target id information in edge embedding"
    )

    parser.add_argument(
        "--node_pred",
        type=str2bool,
        default=True,
        help="whether make predictions based on node representations in GraphAdjModelV2"
    )

    parser.add_argument(
        "--edge_pred",
        type=str2bool,
        default=True,
        help="whether make predictions based on edge representations in GraphAdjModelV2"
    )


def add_eval_config(parser):
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="MSE",
        choices=["MAE", "MSE", "SMSE", "AUC"],
        help="evaluation metric (MAE, MSE, SMSE, AUC)"
    )


def add_train_config(parser):
    parser.add_argument(
        "--bp_loss",
        type=str,
        default="MSE",
        choices=["MAE", "MSE", "SMSE"],
        help="backpropagation loss (MAE, MSE, SMSE)"
    )

    parser.add_argument(
        "--neg_pred_slp",
        type=str2value,
        default="anneal_cosine$1.0$0.01",
        help="slope for negative (0, 0.01, linear$a$b, logistic$a$b, cosine$a$b, cyclical_X$a$b, anneal_X$a$b)"
    )

    parser.add_argument(
        "--match_weights",
        type=str,
        default="none",
        choices=["none", "node", "edge", "node,edge"],
        help="auxiliary matching weights for EdgeSeq and Graph"
    )

    parser.add_argument(
        "--match_loss_w",
        type=str2value,
        default=1.0,
        help="auxiliary matching loss weights (0, 0.01, linear$a$b, cosine$a$b, cyclical_X$a$b, anneal_X$a$b)"
    )

    parser.add_argument(
        "--match_reg_w",
        type=str2value,
        default="anneal_cosine$0.01$0.0",
        help="auxiliary l2 regularizer for predictions (0, 1e-6, linear$a$b, cosine$a$b, cyclical_X$a$b, anneal_X$a$b)"
    )

    parser.add_argument(
        "--rep_reg_w",
        type=str2value,
        default=1e-5,
        help="auxiliary l2 regularizer for representations (0, 1e-5, linear$a$b, cosine$a$b, cyclical_X$a$b, anneal_X$a$b)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_with_warmup_and_restart",
        choices=supported_schedulers.keys(),
        help="learning rate shceduler (constant, linear, cosine, cosine_with_warmup, cosine_with_warmup_and_restart)"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay"
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=8.0,
        help="max gradient norm for clipping"
    )

    parser.add_argument(
        "--early_stop_rounds",
        type=int,
        default=10,
        help="tolerance rounds for early stopping"
    )

    parser.add_argument(
        "--load_model_dir",
        type=str,
        default="",
        help="model path to start finetuning"
    )

    parser.add_argument(
        "--train_epochs",
        type=int,
        default=100,
        help="epochs for training"
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=-1,
        help="batch size for training"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=-1,
        help="batch size for evaluation"
    )

    parser.add_argument(
        "--train_grad_steps",
        type=int,
        default=-1,
        help="steps for update gradients"
    )

    parser.add_argument(
        "--train_log_steps",
        type=int,
        default=-1,
        help="steps for logging"
    )


def add_data_config(parser):
    parser.add_argument(
        "--pattern_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--graph_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--save_data_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--max_npv",
        type=int,
        default=-1,
        help="max number of pattern vertices"
    )

    parser.add_argument(
        "--max_npe",
        type=int,
        default=-1,
        help="max number of pattern edges"
    )

    parser.add_argument(
        "--max_npvl",
        type=int,
        default=-1,
        help="max number of pattern vertex labels"
    )

    parser.add_argument(
        "--max_npel",
        type=int,
        default=-1,
        help="max number of pattern edge labels"
    )

    parser.add_argument(
        "--max_ngv",
        type=int,
        default=-1,
        help="max number of graph vertices"
    )

    parser.add_argument(
        "--max_nge",
        type=int,
        default=-1,
        help="max number of graph edges"
    )

    parser.add_argument(
        "--max_ngvl",
        type=int,
        default=-1,
        help="max number of graph vertex labels"
    )

    parser.add_argument(
        "--max_ngel",
        type=int,
        default=-1,
        help="max number of graph edge labels"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="how many training data are used to train models"
    )

    parser.add_argument(
        "--add_rev",
        type=str2bool,
        default=False,
        help="whether to add reversed edges"
    )

    parser.add_argument(
        "--convert_dual",
        type=str2bool,
        default=False,
        help="whether to convert to conjugate graphs"
    )


def add_device_config(parser):
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="gpu device id"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of threads for multiprocessing"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed"
    )


def get_train_config():
    parser = argparse.ArgumentParser()

    add_device_config(parser)
    add_model_config(parser)
    add_eval_config(parser)
    add_train_config(parser)
    add_data_config(parser)

    args = parser.parse_args()

    return vars(args)


def get_eval_config():
    parser = argparse.ArgumentParser()

    add_device_config(parser)
    add_eval_config(parser)

    parser.add_argument(
        "--pattern_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--graph_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--save_data_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=-1,
        help="batch size for evaluation"
    )

    parser.add_argument(
        "--load_model_dir",
        type=str,
        default="",
        help="model path to start finetuning"
    )

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    from pprint import pprint
    pprint(get_mutag_config())
