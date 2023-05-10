**Mitigating Gender Bias in Neural Language Models**

This repository contains the code and resources for a research project that aims to mitigate gender bias in neural language models by creating a non-gendered dataset and employing it to train and fine-tune multiple HuggingFace models, including GPT-2, BERT, BART, and T5.

**Project Overview**

The project consists of the following key components:

1. Data scraping from the UK Government National Careers Service website
2. Data pre-processing and preparation for training and fine-tuning
3. Training a language model from scratch
4. Fine-tuning pre-trained models (GPT-2, BERT, BART, and T5)
5. Evaluating model bias using toxicity and regard metrics

**Repository Structure**

The repository is organized into several Jupyter notebooks, each focusing on a specific aspect of the project:

1. data_scraping.ipynb: Code for scraping job data from the UK Government National Careers Service website and storing the output in **careers.csv**.
2. data_preprocessing.ipynb: Code for pre-processing the data for training and fine-tuning the language models. The output files are **careers_single.csv** (for training the model from scratch and fine-tuning GPT-2 and T5), **careers_masked.csv** (for fine-tuning BERT), **and bart_masked.csv** (for fine-tuning BART).
3. data_categorise.ipynb: Code for categorizing data into male and female categories and storing the output in **careers_with_gender.csv**.
4. careers_model.ipynb: Code for creating a language model from scratch with **careers_single.csv**.
5. fine-tuned-gpt2.ipynb: Code for fine-tuning the GPT-2 model with **careers_single.csv**.
6. fine-tuned-bert.ipynb: Code for fine-tuning the BERT model with **careers_masked.csv**.
7. fine-tuned-bart.ipynb: Code for fine-tuning the BART model with **bart_masked.csv**.
8. fine-tuned-t5.ipynb: Code for fine-tuning the T5 model with **careers_single.csv**.
9. bias-evaluation-gpt2.ipynb: Code for evaluating bias in the GPT-2 model using toxicity and regard metrics.
10. bias-evaluation-BERT.ipynb: Code for evaluating bias in the BERT model using toxicity and regard metrics.
11. bias-evaluation-BART.ipynb: Code for evaluating bias in the BART model using toxicity and regard metrics.
12. bias-evaluation-t5.ipynb: Code for evaluating bias in the T5 model using toxicity and regard metrics.

**Dependencies**

This project requires Python 3.6 or later and the following packages:

HuggingFace Transformers

Pandas

Numpy

Requests

Beautiful Soup

Jupyter



