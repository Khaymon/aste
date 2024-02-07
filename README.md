# Aspect Sentement Triplet Extraction

<image src="wizard.jpeg" alt="Sentiment Wizard" caption="Image is generated via DALL-E 3">


# Experiments
## T5 fine-tuning

| checkpoint              | P      | R      | F1     | recipe                                         | decodding                |
| ----------------------- | ------ | ------ | ------ | ---------------------------------------------- | ------------------------ |
| ai-forever/FRED-T5-1.7B | 0.3424 | 0.3698 | 0.3556 | experiment-01-t5-1.7B-freeze-400               |                          |
| ai-forever/FRED-T5-1.7B | 0.554  | 0.405  | 0.4679 | experiment-03-t5-1.7B-freeze-300-sentences     |                          |
| ai-forever/FRED-T5-1.7B | 0.4399 | 0.5825 | 0.5013 | experiment-04-t5-1.7B-freeze-300-sentences-mvp | 3 orders, without voting |
| ai-forever/FRED-T5-1.7B | 0.612  | 0.4691 | 0.5311 | experiment-04-t5-1.7B-freeze-300-sentences-mvp | 3 orders, 2 quorum       |
| ai-forever/FRED-T5-1.7B | 0.7231 | 0.3446 | 0.4668 | experiment-04-t5-1.7B-freeze-300-sentences-mvp | 3 orders, 3 quorum       |
|                         |        |        |        |                                                |                          |