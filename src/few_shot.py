import os

import numpy as np

from src.models.flyp_loss_few_shot import flyp_loss_few_shot
from src.models.modeling import CLIPEncoder
from src.args import parse_arguments
import logging


def main(args):
    assert args.k in [4, 16, 32], 'please specify correct k'
    
    ###logging##################################################################
    os.makedirs(args.save + args.exp_name, exist_ok=True)
    args.save = args.save + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    logger.info(args)
    assert args.save is not None, 'Please provide a path to store models'
    ###logging##################################################################

    val_accs = []
    test_accs = []

    for run_iter in range(50):
        logger.info(
            f'------------------ Running iteration {run_iter} -------------------'
        )
        # Build and save zero-shot model
        clip_encoder = CLIPEncoder(args, keep_lang=True)
        classification_head = None

        val_acc, test_acc = flyp_loss_few_shot(args, clip_encoder,
                                               classification_head, logger)
        logger.info(f'Val {val_acc} {test_acc}')
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir + '/stats_final.txt', 'w') as f:
        f.write(f'Val: {round(np.mean(val_accs),4)}\n')
        f.write(f'Test: {round(np.mean(test_accs),4)}\n')
        f.write(f'ValDev: {round(np.std(val_accs),4)}\n')
        f.write(f'TestDev: {round(np.std(test_accs),4)}\n')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
