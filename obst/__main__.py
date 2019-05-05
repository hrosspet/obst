import sys
import fire
import pdb, traceback, sys

from obst.logs import generate_run_id, prepare_logging
from obst.eval import Eval
from obst.config import CONFIG

TRAINING_STEPS = CONFIG['TRAINING_STEPS']
TEST_STEPS = CONFIG['TEST_STEPS']

def main(verbosity='INFO', loglevel='DEBUG', gitdir='.git'):
    try:
        world = CONFIG['WORLD']['constructor'](**CONFIG['WORLD']['ctor_params'])
        agent = CONFIG['AGENT']['constructor'](dims=CONFIG['WORLD']['dims'], repr_model=CONFIG['WORLD']['repr_model'], **CONFIG['AGENT']['ctor_params'])
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

        print('evaluation.train():', evaluation.train())
        # print('evaluation.test():', evaluation.test())
    except KeyboardInterrupt:
        logger.warning("Terminated by user.")
    except SystemExit:
        logger.info("Finished.")
    except:
        # Trigger debugger on exception     # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    finally:
        if world.__class__.__name__ == 'ObstTowerWorld':
            world.env.close()

    logger.info("######### FINISHED #########")


if __name__ == "__main__":
    fire.Fire(main)
