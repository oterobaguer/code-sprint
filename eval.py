import argparse

from dival import Reconstructor
from dival import TaskTable, DataPairs
from dival import get_standard_dataset
from dival.measure import SSIM, PSNR

from learnedpd import LearnedPDReconstructor


class EnsembleReconstructor(Reconstructor):

    def __init__(self, *reconstructors, **kwargs):
        super().__init__(**kwargs)
        self.reconstructors = reconstructors

    def _reconstruct(self, observation):
        reconstructions = [r.reconstruct(observation)
                           for r in self.reconstructors]
        return sum(reconstructions)/len(reconstructions)


def main():
    """Main function"""

    dataset = get_standard_dataset('lodopab')
    test_data = dataset.get_data_pairs('test', 100)
    n_ensemble = 10
    # load reconstructor
    reconstructors = []
    for i in range(n_ensemble):
        reconstructor = LearnedPDReconstructor(dataset.ray_trafo)
        reconstructor.load_hyper_params('params')
        reconstructor.load_learned_params('best-model-%d' % i)
        reconstructors.append(reconstructor)

    ensemble = EnsembleReconstructor(*reconstructors)

    task_table = TaskTable()
    task_table.append(
        reconstructor=ensemble,
        measures=[PSNR, SSIM],
        test_data=test_data,
        options={'skip_training': True}
    )
    task_table.run()

    print(task_table.results.to_string(show_columns=['misc']))


if __name__ == '__main__':
    main()
