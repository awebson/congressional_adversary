import argparse
from pathlib import Path

import torch

from data import GroundedWord
from models.ideal_grounded import Decomposer, Recomposer
from models.proxy_grounded import ProxyGroundedDecomposer, ProxyGroundedRecomposer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', action='store', type=Path)
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = in_dir
    device = 'cuda:0'

    checkpoints = list(in_dir.glob('epoch*.pt'))

    # rand_path: Path = Path('../../data/ready/CR_topic_context3/eval_words_random.txt')
    # dev_path: Path = Path('../../data/ready/CR_topic_context3/0.7partisan_dev_words.txt')
    # test_path: Path = Path('../../data/ready/CR_topic_context3/0.7partisan_test_words.txt')

    rand_path: Path = Path('../../data/ready/CR_bill_context3/eval_words_random.txt')
    dev_path: Path = Path('../../data/ready/CR_bill_context3/0.7partisan_dev_words.txt')
    test_path: Path = Path('../../data/ready/CR_bill_context3/0.7partisan_test_words.txt')



    with open(dev_path) as file:
        dev_words = [word for word in file]
    with open(test_path) as file:
        test_words = [word for word in file]
    with open(rand_path) as file:
        rand_words = [word for word in file]



    with open(test_path) as file:
        test_ids = torch.tensor(
            [model.word_to_id[word.strip()] for word in file],
            device=device)
    with open(rand_path) as file:
        rand_ids = torch.tensor(
            [model.word_to_id[word.strip()] for word in file],
            # if word.strip() in model.word_to_id],
            device=device)

    for in_path in checkpoints:
        model = torch.load(in_path, map_location=device)
        print(in_path)

        dev_ids = torch.tensor(
            [model.word_to_id[word.strip()] for word in dev_words],
            device=device)
        test_ids = torch.tensor(
            [model.word_to_id[word.strip()] for word in test_words],
            device=device)
        rand_ids = torch.tensor(
            [model.word_to_id[word.strip()] for word in rand_words],
            device=device)

        dev_Hd, dev_Hc = model.deno_space.homogeneity(dev_ids)
        test_Hd, test_Hc = model.deno_space.homogeneity(test_ids)
        rand_Hd, rand_Hc = model.deno_space.homogeneity(rand_ids)
        model.PE_homogeneity = {
            'dev Hd': dev_Hd,
            'dev Hc': dev_Hc,
            'test Hd': test_Hd,
            'test Hc': test_Hc,
            'rand Hd': rand_Hd,
            'rand Hc': rand_Hc,
        }

    DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.dev_ids)


if __name__ == "__main__":
    main()
