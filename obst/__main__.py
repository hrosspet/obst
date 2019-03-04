import fire
from obst.logs import generate_run_id, prepare_logging


def main(verbosity='INFO', loglevel='INFO', gitdir='.git'):

    ####
    some_func_name = '__some_func_name__'
    ####

    global RUN_ID
    RUN_ID = generate_run_id(gitdir, some_func_name)
    logger = prepare_logging(verbosity, RUN_ID, loglevel)

    logger.info('verbosity: %s, loglevel: %s, gitdir: %s', verbosity, loglevel, gitdir)
    logger.info("######### STARTING #########")
    logger.info('run_id: %s', RUN_ID)

    try:
        print('run eval')
    except KeyboardInterrupt:
        logger.warning("Terminated by user.")
    except SystemExit:
        logger.info("Finished.")
    logger.info("######### FINISHED #########")


if __name__ == "__main__":
    fire.Fire(main)