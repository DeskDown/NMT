# Exploring the limits of Mix-training for NMT task on low resource languages.

Our project showcases the impressive utility of multi-parallel corpora for transfer learning in a one-to-many low-resource neural machine translation (NMT) setting.
Training procedures are highly motivated by Raj et al. 2019 paper **Exploiting Multilingualism through Multistage Fine-Tuning for Low-Resource Neural Machine Translation**.


## Datasets

We used ALT dataset which was introduced by **Asian Language Treebank (ALT) Project.**

The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages.

We used translation of 9 different asian languages in our work. These languages are:

1. Filipino, (hi)
2. Hindi, (hi)
3. Bahasa Indonesia, (id)
4. Japanese, (ja)
5. Khmer, (khm)
6. Malay, (ms)
7. Myanmar (Burmese),
8. Thai, (th)
9. Vietnamese, (vi)
10. Chinese, (zh)

For mix training purposes, we used the opus dataset (en-ja or en-zh) which was used to pretrain the text-to-text transformer model.

Even when the helping target language is not one of the target languages of our concern, the multistage fine-tuning process provided 3–9 BLEU score gains over a simple one-to-one model.

## Tokenizers

Tokenizing the low resource asian languages is also a critical task. These languages are very different in terms of their syntax and might not follow the same *space* or *punctuation* rules as other languages. On top of that, every Hugging face translation model comes with its own tokenizer, which was used during pre-training of that model. Using any other model will completely demolish the pre-training knowledge of NMT model. To overcome this hurdle, we used a *teacher* tokenizer to tokenize ALT dataset training sentences. The tokens were later saved and added to the *student* tokenizer's vocabulary. As the teacher we used *facebook/mbart-large-50-one-to-many-mmt* from HF Hub. The *student* tokenizer was always the one which was used to pretrain the desired model. 

Our implementation is available [here](./make_tokens.py)

## Training

In our experiments we used 3 different training techniques:

1. **Pure Finetuning the pretrained model**

   For each target language in ALT dataset, a HF pretrained model is finetuned over that language.
2. **Mix Finetuning the pretrained model**

   A HF model is finetuned on all languages from ALT dataset + 20K sentences from pretrainind dataset corpus. Model is saved for future use.
3. **Pure Finetuning the Mix Finetuned model**

   For each language in ALT dataset, we load the model trained in step 2 and perform purefinetuning using selected language.

## Results

## Acknowledgements

 - [Exploiting Multilingualism through Multistage Fine-Tuning for Low-Resource Neural Machine Translation](https://aclanthology.org/D19-1146.pdf)
 - [Marian: Fast Neural Machine Translation](https://huggingface.co/docs/transformers/model_doc/marian)
 - [mBART-50 many to many multilingual machine translation](https://arxiv.org/abs/2008.00401)
