import os
import torch
import matplotlib.pyplot as plt


# TODO: make the following changes to utils.py and attacks.py:
'''
# in utils.py

# add this to parse_args():
parser.add_argument('--run_name', default='', help='name of run for graphs. cannot have any "_" in it!!! (default: "")')

# add this to the end of compute_output_dir():
args.output_dir += '_' + str(args.run_name)

# it should look like this:
            args.output_dir += "/eps_" + str(args.eps).replace('.', '_') + \
                               "_attack_iter_" + str(args.attack_k) + \
                               "_alpha_" + str(args.alpha).replace('.', '_')
----->      args.output_dir += '_' + str(args.run_name)
            if not isdir(args.output_dir): 
                mkdir(args.output_dir)
'''

'''
# in run_attacks.py

import os
from attacks import Const

# after this line:
best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device)

# insert:

torch.save(all_loss_list, os.path.join(args.output_dir, 'all_loss_list.pt'))
components_listname = args.output_dir.split('/')
listname = ''
for component in components_listname:
    listname += '_' + component
listname += '.pt'
listname = listname.split('opt_whole_trajectory')[-1]  # cutting down listname length this way is not elegant, but it works for now. alternatively you can save only run name, but this way custom filtration might be added in the future
list_path = os.path.join("results/loss_lists", listname)
if not isinstance(attack, Const):
    print(f'saving all_loss_list to {list_path}')
    torch.save(all_loss_list, list_path)

'''


def cumul_sum_loss_from_list(loss_list):
    """
    :param loss_list: a list(epochs) of lists(trajectories) of lists(frames) of losses.
    :return: a list of the cumulative sum of the losses per epoch. this is what is printed in PGD.perturb during evaluation.
    """

    return [sum([sum(trajectory_losses) for trajectory_losses in epoch_losses]) for epoch_losses in loss_list]


def sum_loss_from_list(list3):
    """
    :param list3: a list(epochs) of lists(trajectories) of lists(frames) of losses.
    in the code:
    # list3 is trajectory losses per epoch
    # list2 is losses per trajectory
    # list1 is loss per frame of a trajectory
    :return: a list of the sum of the last frame losses per epoch. this is what I think we are going to be rated on.
    """

    last_frame_loss = [[list1[-1] for list1 in list2] for list2 in list3]
    sum_list = [sum(l) for l in last_frame_loss]
    return sum_list


def compare_results(loss_list_dir, agg=cumul_sum_loss_from_list):
    """
    :param loss_list_dir:  directory of loss_list.pt files
    :param agg: function to aggregate the loss lists. can be either cumul_sum_loss_from_list or sum_loss_from_list.
    """
    def extract_label(filename):
        components = filename.split('_')
        return components[-1]

    for filename in os.listdir(loss_list_dir):
        print(filename)
        f = os.path.join(loss_list_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            loss_list = torch.load(f)
            loss_per_epoch = agg(loss_list)
            plt.plot(loss_per_epoch, label=extract_label(filename))
    plt.legend(loc='lower right', fontsize=6)
    print("showing")
    plt.show()


if __name__ == '__main__':
    # to actually see the plots, this script should be run from pycharm(or any other ide?) with remote interpreter
    # see https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html#ssh
    compare_results("results/loss_lists")
