
import argparse
from configs import cfg
from sokoban.soko_pap import *
from utils.get_sokoban import *
from utils.display_utils import *
from algorithms.model_free.deep_q_learning import *
# GetSokoban()


#############################################################
# The main file - run the chosen Algorithm to solve Sokoban #
#############################################################


def parse_args():
    # create parser
    parser = argparse.ArgumentParser(description='Solve Sokoban')
    # add the cfg file as an argument
    parser.add_argument("-config_file", default="config.yaml", help="path to config file", type=str)
    # parse arguments
    args, unknown = parser.parse_known_args()
    # override default configurations
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(unknown)
    cfg.freeze()

    return args


def SetEnv(sokoban_cfg, max_steps):
    if sokoban_cfg.FIX 
        if 1 == sokoban_cfg.NUM_BOXES:
            random.seed(2)
        else:
            random.seed(6)
    env = PushAndPullSokobanEnv(dim_room=(sokoban_cfg.DIM, sokoban_cfg.DIM),
                                num_boxes=sokoban_cfg.NUM_BOXES, max_steps=max_steps)
    return env


def SetAlgorithm(algo_cfg, sokoban_cfg, env):
    algo = DQN(env, algo_cfg.EPSILON_GREEDY, algo_cfg.DECAY, algo_cfg.DISCOUNT_FACTOR,
               algo_cfg.CHANGE_REWARD)
    if 1 == sokoban_cfg.NUM_BOXES:
        algo.policy.load_state_dict(torch.load('model1.pth', map_location=torch.device('cpu')))
    else:
        algo.policy.load_state_dict(torch.load('path_to_your_model.pth', map_location=torch.device('cpu')))
    algo.policy.eval()
    return algo


def play_video(algo):
    name = f'Sokoban_{cfg.SOKOBAN.DIM}x{cfg.SOKOBAN.DIM}_{cfg.SOKOBAN.NUM_BOXES}_boxes'
    DisplayVideo(algo.test(name))


def main():
    args = parse_args()
    env = SetEnv(cfg.SOKOBAN, cfg.ALGORITHM.NUM_STEPS)
    algo = SetAlgorithm(cfg.ALGORITHM, cfg.SOKOBAN, env)
    # algo = SetAlgorithmWithTraining(cfg.ALGORITHM, env) # run with training
    play_video(algo)


if __name__ == "__main__":
    main()