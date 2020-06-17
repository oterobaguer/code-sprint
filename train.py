import argparse
import random

from dival import TaskTable
from dival import Dataset
from dival import get_standard_dataset
from dival.measure import PSNR, SSIM

from learnedpd import LearnedPDReconstructor


class RandomSampleDataset(Dataset):
    """Dataset that allows to use cached elements of a dataset
    """

    def __init__(self, dataset, size_part=1.0, seed=0):

        super().__init__(space=(dataset.ray_trafo.range,
                                dataset.ray_trafo.domain))
        random.seed(seed)
        self.dataset = dataset

        self.train_len = int(size_part * self.dataset.train_len)
        self.train_len = max(1, self.train_len)
        self.validation_len = int(size_part * self.dataset.validation_len)
        self.validation_len = max(1, self.validation_len)

        self.idx = {'train': list(range(self.train_len)),
                    'validation': list(range(self.validation_len))}

        random.shuffle(self.idx['train'])
        random.shuffle(self.idx['validation'])

        self.random_access = True

    def get_sample(self, index, part='train', out=None):

        if index >= self.get_len(part):
            raise IndexError(
                "index {:d} out of range for dataset part '{}' (len: {:d})"
                .format(index, part, self.get_len(part)))

        index = self.idx[part][index]
        return self.dataset.get_sample(index, part=part)


def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=None)
    return parser


def main():
    # load data
    options = get_parser().parse_args()

    dataset = get_standard_dataset('lodopab')
    test_data = dataset.get_data_pairs('validation')
    ray_trafo = dataset.ray_trafo

    reduced_dataset = RandomSampleDataset(
        dataset, size_part=0.1, seed=options.seed)

    reconstructor = LearnedPDReconstructor(
        ray_trafo=ray_trafo,
        num_workers=8)
    reconstructor.load_hyper_params('params')

    reconstructor.save_best_learned_params_path = 'best-model-{}'.format(
        options.seed)
    reconstructor.log_dir = options.log_dir

    # create a Dival task table and run it
    task_table = TaskTable()
    task_table.append(
        reconstructor=reconstructor,
        measures=[PSNR, SSIM],
        test_data=test_data,
        dataset=reduced_dataset,
        hyper_param_choices=[reconstructor.hyper_params]
    )
    results = task_table.run()

    # save report
    save_results_table(results, full_name)


if __name__ == '__main__':
    main()
