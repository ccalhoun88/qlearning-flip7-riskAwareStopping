import logging

# Training logger — metrics and snapshots
training_logger = logging.getLogger("flip7.training")

def log_training(msg: str) -> None:
    training_logger.info(msg)
