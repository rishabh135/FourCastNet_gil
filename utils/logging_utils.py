import logging
import os

_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the date to get the day and month
day_month = now.strftime("%B_%d_")


def config_logger(log_level=logging.INFO):
    logging.basicConfig(
        filename=f"./FourCastNet_{day_month}_logging.log",
        format=_format,
        level=log_level,
    )


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename=f"./FourCastNet_{day_month}_logging.log",
):
    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    import subprocess
    import torch
    logging.info("--------------- Versions ---------------")
    #   logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
    #   logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")
