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

    # load reconstructor
    reconstructor1 = LearnedPDReconstructor(dataset.ray_trafo)
    reconstructor1.load_hyper_params('params')
    reconstructor1.load_learned_params('best-model-0')

    reconstructor2 = LearnedPDReconstructor(dataset.ray_trafo)
    reconstructor2.load_hyper_params('params')
    reconstructor2.load_learned_params('best-model-1')

    ensemble = EnsembleReconstructor(reconstructor1, reconstructor2)

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
