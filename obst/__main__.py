import fire
from obst.logging import generate_run_id, prepare_logging


def main(verbosity='INFO', loglevel='INFO', gitdir='.git'):

    ####
    some_func_name = '__some_func_name__'
    ####

    global RUN_ID
    RUN_ID = generate_run_id(gitdir, some_func_name)
    logger = prepare_logging(verbosity, RUN_ID, loglevel)

    logger.info('verbosity:', verbosity, 'loglevel:', loglevel, 'gitdir:', gitdir)
    logger.info("######### STARTING #########")
    logger.info('run_id: %s', RUN_ID)
    # logger.debug('Received command line arguments: \n{}'.format(args))

    try:
        # args.func(args)
        print('try')
    except KeyboardInterrupt:
        logger.warning("Terminated by user.")
    except SystemExit:
        logger.info("Finished.")
    logger.info("######### FINISHED #########")


if __name__ == "__main__":
    fire.Fire(main)