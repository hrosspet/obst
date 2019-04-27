import sys
import os
import logging
from datetime import datetime
from time import gmtime
from subprocess import check_output

from obst.config import CONFIG

TIME_FORMAT = CONFIG['TIME_FORMAT']
RUN_ID = datetime.strftime(datetime.utcnow(), '%Y%m%d%H%M%S')

def get_git_dir_hash(gitdir):
    return check_output(['git',
                         '--git-dir={}'.format(gitdir),
                         'rev-parse',
                         '--short',
                         'HEAD']).decode().strip()


def generate_run_id(gitdir, run_func):
    # commit_hash = get_git_dir_hash(gitdir)
    # return '{}-{}-{}'.format(RUN_ID, commit_hash, run_func)   # Can't do this because Docker image doesn't have git installed and .git directory isn't present
    return '{}-{}'.format(RUN_ID, run_func)


class InjectRunID(logging.Filter):
    def __init__(self, run_id):
        self.run_id = run_id
        super().__init__()

    def filter(self, record):
        record.run_id = self.run_id
        return True


def prepare_logging(level, run_id, loglevel):
    logger = logging.getLogger()
    logging.Formatter.converter = gmtime
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s - %(message)s",
                                  CONFIG['TIME_FORMAT'])
    injecter = InjectRunID(run_id)

    logger.setLevel('DEBUG')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.addFilter(injecter)
    ch.setLevel(level)
    logger.addHandler(ch)

    if loglevel:
        os.makedirs('logs/', exist_ok=True)
        lh = logging.FileHandler('logs/{}.log'.format(run_id))
        lh.setFormatter(formatter)
        lh.addFilter(injecter)
        lh.setLevel(loglevel)
        logger.addHandler(lh)
        logger.info('Logging %s into file %s', loglevel, run_id)

    return logger
