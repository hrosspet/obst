import fire
from obst.logs import generate_run_id, prepare_logging
from obst.eval import Eval
from obst.config import CONFIG

TRAINING_STEPS = CONFIG['TRAINING_STEPS']
TEST_STEPS = CONFIG['TEST_STEPS']


def main(verbosity='INFO', loglevel='INFO', gitdir='.git'):

    ####
    world = CONFIG['WORLD']['CONSTRUCTOR'](**CONFIG['WORLD']['PARAMS'])
    agent = CONFIG['AGENT']['CONSTRUCTOR'](**CONFIG['AGENT']['PARAMS'])
    evaluation = Eval(world, agent, training_steps=TRAINING_STEPS, test_steps=TEST_STEPS, vis_steps=CONFIG['VISUALIZE_STEPS'])

    run_name = '_'.join([world.__class__.__name__, agent.__class__.__name__])
    print('\nWorld:\t{}\nAgent:\t{}\n'.format(world.__class__.__name__, agent.__class__.__name__))
    ####

    global RUN_ID
    RUN_ID = generate_run_id(gitdir, run_name)
    logger = prepare_logging(verbosity, RUN_ID, loglevel)

    logger.info('verbosity: %s, loglevel: %s, gitdir: %s', verbosity, loglevel, gitdir)
    logger.info("######### STARTING #########")
    logger.info('run_id: %s', RUN_ID)

    try:
        print('evaluation.train():', evaluation.train())
        # print('evaluation.test():', evaluation.test())
    except KeyboardInterrupt:
        logger.warning("Terminated by user.")
    except SystemExit:
        logger.info("Finished.")
    # finally:
    #     agent.save_weights('weights/')

    logger.info("######### FINISHED #########")


if __name__ == "__main__":
    fire.Fire(main)
