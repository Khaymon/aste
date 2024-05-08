# Aspect Sentement Triplet Extraction

<image src="wizard.jpeg" alt="Sentiment Wizard" caption="Image is generated via DALL-E 3">


# Experiments
## T5 fine-tuning

| checkpoint              | P      | R      | F1     | recipe                                                            | decodding                       |
| ----------------------- | ------ | ------ | ------ | ----------------------------------------------------------------- | ------------------------------- |
| ai-forever/FRED-T5-1.7B | 0.3424 | 0.3698 | 0.3556 | experiment-01-t5-1.7B-freeze-400                                  |                                 |
| ai-forever/FRED-T5-1.7B | 0.554  | 0.405  | 0.4679 | experiment-03-t5-1.7B-freeze-300-sentences                        |                                 |
| ai-forever/FRED-T5-1.7B | 0.4399 | 0.5825 | 0.5013 | experiment-04-t5-1.7B-freeze-300-sentences-mvp                    | 3 orders, without voting        |
| ai-forever/FRED-T5-1.7B | 0.612  | 0.4691 | 0.5311 | experiment-04-t5-1.7B-freeze-300-sentences-mvp                    | 3 orders, 2 quorum              |
| ai-forever/FRED-T5-1.7B | 0.7231 | 0.3446 | 0.4668 | experiment-04-t5-1.7B-freeze-300-sentences-mvp                    | 3 orders, 3 quorum              |
| ai-forever/FRED-T5-1.7B | 0.6379 | 0.4846 | 0.5507 | experiment-04-t5-1.7B-freeze-300-sentences-mvp                    | 6 orders, 3 quorum, levenshtein |
| ai-forever/FRED-T5-1.7B | 0.5623 | 0.4695 | 0.5117 | experiment-04-t5-1.7B-freeze-300-sentences-mvp                    | 6 orders, 3 quorum              |
| ai-forever/FRED-T5-1.7B | 0.6311 | 0.4211 | 0.5051 | experiment-05-t5-1.7B-freeze-300-sentences-mvp-pos-tags           | 6 orders, 3 quorum, levenshtein |
| ai-forever/FRED-T5-1.7B | 0.6532 | 0.4513 | 0.5337 | experiment-05-t5-1.7B-freeze-300-sentences-mvp-pos-tags-blacklist | 6 orders, 3 quorum, levenshtein |
|                         |        |        |        |                                                                   |                                 |
Select with tree -- | ai-forever/FRED-T5-1.7B | 0.6698 | 0.4712 | 0.5532 |
LoRA -- Precision: 0.48034934497816595, Recall: 0.49736247174076864, F1: 0.48870788596815995
Precision: 0.5763016157989228, Recall: 0.48379804069329313, F1: 0.5260139287177388