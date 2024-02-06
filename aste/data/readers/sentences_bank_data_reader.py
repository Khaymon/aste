import ast
import pathlib
import typing as T

from razdel import sentenize

import aste.data.common as data_common_lib

from .base_data_reader import BaseDataReader


class SentenceBankDataReader(BaseDataReader):
    TARGET_SEPARATOR = "####"
    TOKENS_SEPARATOR = ' '

    @staticmethod
    def from_file(path: pathlib.Path, train: bool = True) -> T.List[data_common_lib.SampleData]:
        with open(path, 'r') as input_file:
            lines = input_file.readlines()

        data = []
        for sample_id, line in enumerate(lines):
            text, aspects = line.split(SentenceBankDataReader.TARGET_SEPARATOR)
            sentences = list(sentenize(text))
            aspects = ast.literal_eval(aspects)
            aspects = sorted(aspects, key=lambda x: min(x[0] + x[1]))  # Sort aspects by min of aspect and opinion idx

            aspect_idx = 0
            sentence_idx = 0
            tokens_viewed = 0

            sentence_aspects = []
            while aspect_idx < len(aspects) and sentence_idx < len(sentences):
                sentence = sentences[sentence_idx].text
                aspect_ids, opinion_ids, polarity = aspects[aspect_idx]

                sentence_tokens = len(sentence.split(SentenceBankDataReader.TOKENS_SEPARATOR))
                if max(aspect_ids + opinion_ids) < tokens_viewed + sentence_tokens and min(aspect_ids + opinion_ids) >= tokens_viewed:
                    sentence_aspects.append(
                        data_common_lib.AspectData(
                            aspect=SentenceBankDataReader._str_from_ids(text, aspect_ids),
                            opinion=SentenceBankDataReader._str_from_ids(text, opinion_ids),
                            polarity=polarity,
                            aspect_ids=aspect_ids,
                            opinion_ids=opinion_ids,
                        )
                    )
                    aspect_idx += 1
                elif max(aspect_ids + opinion_ids) < tokens_viewed:
                    aspect_idx += 1
                else:
                    data.append(data_common_lib.SampleData(sample_id=sample_id, text=sentence, aspects=sentence_aspects))
                    sentence_aspects = []
                    sentence_idx += 1
                    tokens_viewed += sentence_tokens
            
            while sentence_idx < len(sentences):
                data.append(data_common_lib.SampleData(sample_id=sample_id, text=sentences[sentence_idx].text, aspects=sentence_aspects))
                sentence_aspects = []
                sentence_idx += 1

        return data
