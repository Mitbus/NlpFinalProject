# NlpFinalProject

Implements the final project for the NLP course

# Contents

1. `data` - training data
1.1. `gsm8k-ru` - contains the original and post-processed SGM8k-ru data
1.1.1. `postprocessed` - data after postprocessing
1.2. `MAWPS` - contains raw, post-processed and intermediate MAWPS data
1.2.1. `dataset.csv` - the final result of post-processing of our dataset
1.2.2. `MAWPS_train.csv` and `MAWPS_validation.csv` - original MAWPS dataset
1.2.3. All remaining directories contain intermediate results
2. `generations` - results of generating solutions on a test dataset
2.1. `brackets` - calculator format with brackets
2.2. `calculator` - calculator format with tags
2.3. `denoisers` - experiment with denoisers
2.4. `pretrain` - experiment with pretraining
2.5. `saiga` - experiment with retrained saiga Mistral
3. `gptTransform` - a utility for generating a dataset.
3.1. `transform.py` - requests to change the dataset (generating a solution and formatting a response)
3.2. `join.py` - joining datasets script
4. `models` - trained models
5. `train_saiga` - tools for training saiga/Mistral7B
6. `gsm8k.ipynb` - pre-processing of the SGM8k-ru dataset
7. `test_saiga.ipynb` - running tests for saiga/Mistral7B
8. `test_t5.ipynb` - running tests for FRED-T5
9. `train_t5.ipynb` - FRED-T5 training
