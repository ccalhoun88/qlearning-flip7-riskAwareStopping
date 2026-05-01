# Added logging for the two loggers in the engine and the training
import logging

def setup_logging():
    # Engine game events
    engine_handler = logging.FileHandler("flip7_game_log.txt", mode="w")
    engine_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("flip7.engine").addHandler(engine_handler)
    logging.getLogger("flip7.engine").setLevel(logging.INFO)

    # Training metrics
    training_handler = logging.FileHandler("flip7_training_log.txt", mode="w")
    training_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("flip7.training").addHandler(training_handler)
    logging.getLogger("flip7.training").setLevel(logging.INFO)
