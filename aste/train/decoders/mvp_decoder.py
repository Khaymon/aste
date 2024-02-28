from collections import defaultdict
import pathlib
import typing as T 

from aste.data.common import AspectData
from aste.train.data_providers.datasets import BaseDataset, MVPGenerativeDataset

from .base_decoder import BaseDecoder


class MVPDecoder(BaseDecoder):
    @staticmethod
    def decode(
        file: pathlib.Path,
        dataset_class: T.Type[BaseDataset] = MVPGenerativeDataset,
        votes: int = 2
    ) -> T.Dict[int, T.Set[AspectData]]:
        with open(file, 'r') as input_file:
            lines = input_file.readlines()

        counts = defaultdict(lambda: defaultdict(int))
        for line in lines:
            sample_id, text, prediction = line.split('\t')
            sample_id = int(sample_id)
            counts[sample_id]  # to be sure that all sample_ids are in result

            for predicted_aspect in dataset_class.decode(text=text, prediction=prediction):
                aspect_nearest = [BaseDecoder.nearest_levenshtein_word(text, word) for word in predicted_aspect.aspect.split()]
                opinion_nearest = [BaseDecoder.nearest_levenshtein_word(text, word) for word in predicted_aspect.opinion.split()]

                # if not any(word is None for word in aspect_nearest + opinion_nearest):
                #     predicted_aspect.aspect = ' '.join(aspect_nearest)
                #     predicted_aspect.opinion = ' '.join(opinion_nearest)
                counts[sample_id][predicted_aspect] += 1

        result_predictions = defaultdict(set)
        for sample_id in counts:
            result_predictions[sample_id].update(
                predicted_aspect for predicted_aspect in counts[sample_id]
                if counts[sample_id][predicted_aspect] >= votes
            )

        return result_predictions
