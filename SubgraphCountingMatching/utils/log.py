import logging
import os
import re


def init_logger(log_file=None, log_tag="GOOD LUCK"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter("[%(asctime)s {log_tag}] %(message)s".format(log_tag=log_tag))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    setattr(logger, "prefix_len", len(logger.handlers[0].formatter._fmt))

    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def generate_log_line(data_type, epoch=-1, total_epochs=-1, step=-1, total_steps=-1, **kw):
    line = ["data_type: {:<10s}".format(data_type)]
    if epoch != -1 and total_epochs != -1:
        line.append("epoch: {:0>5d}/{:0>5d}".format(epoch, total_epochs))
    if step != -1 and total_steps != -1:
        line.append("step: {:0>5d}/{:0>5d}".format(step, total_steps))
    for k, v in kw.items():
        if isinstance(v, float):
            line.append("{}: {:8>5.3f}".format(k, v))
        elif isinstance(v, int):
            line.append("{}: {:0>3d}".format(k, v))
        else:
            line.append("{}: {}".format(k, v))
    line = "\t".join(line)
    return line


def generate_best_line(data_type, epoch, total_epochs, **kw):
    line = \
        ["data_type: " + str(data_type)] + \
        ["best %s: %s" % (str(k), str(v)) for k, v in kw.items()] + \
        ["(epoch: %d/%d)" % (epoch, total_epochs)]
    line = "\t".join(line)
    return line


def get_best_epochs(log_file):
    # 0: data type
    # 1: metric name
    # 2: metric value
    # 3: epoch
    regex = re.compile(
        r"data_type:\s+(\w+)\s+best\s+([a-zA-Z0-9\.\-\+\_]+):\s+([a-zA-Z0-9\.\-\+\_]+)\s+\(epoch:\s+(\d+)/\d+\)"
    )
    best_epochs = dict()
    # get the best epoch
    with open(log_file, "r") as f:
        for line in f:
            matched_results = regex.findall(line)
            for matched_result in matched_results:
                if matched_result[1] not in best_epochs:
                    best_epochs[matched_result[1]] = dict()
                best_epochs[matched_result[1]][matched_result[0]] = (int(matched_result[3]), float(matched_result[2]))
    return best_epochs
