import sys, os
import fire
import pdb, traceback

from obst.models import HasWeights
from obst.logs import generate_run_id, prepare_logging
from obst.eval import Eval
from obst.config import CONFIG

def main(verbosity='INFO', loglevel='DEBUG', gitdir='.git'):
    try:
        world = CONFIG['WORLD']['constructor'](**CONFIG['WORLD']['ctor_params'])
        agent = CONFIG['AGENT']['constructor'](dims=CONFIG['WORLD']['dims'], repr_model=CONFIG['WORLD']['repr_model'], **CONFIG['AGENT']['ctor_params'])
        evaluation = Eval(world, agent, CONFIG['INTERVALS'])

        run_name = '_'.join([world.__class__.__name__, agent.__class__.__name__])
        print('\nWorld:\t{}\nAgent:\t{}\n'.format(world.__class__.__name__, agent.__class__.__name__))
        ####

        global RUN_ID
        RUN_ID = generate_run_id(gitdir, run_name)
        logger = prepare_logging(verbosity, RUN_ID, loglevel)

        logger.info('verbosity: %s, loglevel: %s, gitdir: %s', verbosity, loglevel, gitdir)
        logger.info("######### STARTING #########")
        logger.info('run_id: %s', RUN_ID)

        if 'WEIGHTS_DIR' in CONFIG:
            weights_dir = CONFIG['WEIGHTS_DIR']
            if os.path.isdir(weights_dir):                                      # If the weights dir doesn't exist yet, nothing will be loaded but it'll be created when saving.
                logger.info('Loading weights from {}.'.format(weights_dir))
                agent.load_weights_from_dir(weights_dir)

        evaluation.eval(CONFIG['INTERVALS']['abort_at'])
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
        # Close the world
        world.close()

        # Save the weights if the agent supports it
        if isinstance(agent, HasWeights):
            if 'WEIGHTS_DIR' in CONFIG:
                weights_dir = CONFIG['WEIGHTS_DIR']

                if not os.path.isdir(weights_dir):  # Create the weights dir if it doesn't exist
                    os.makedirs(weights_dir)

                logger.info('Saving weights to {}.'.format(weights_dir))
                agent.save_weights_to_dir(weights_dir)

    logger.info("######### FINISHED #########")


if __name__ == "__main__":
    fire.Fire(main)
