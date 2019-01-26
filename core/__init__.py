# See LICENSE
import logging
import warnings
warnings.filterwarnings("ignore") # ignore messy numpy warnings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')