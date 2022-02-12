from tqdm.auto import tqdm
from transformers import MarianTokenizer, MBart50TokenizerFast, utils
from datasets import Dataset, load_dataset
utils.logging.set_verbosity(50)


def update_tokenizer(
    teacher,
    student,
    sentences,
    CREATE_TOKENS_FILE=False,
    TOKENS_FILE_NAME='Tokens.txt'
):
    """Updated the marian tokenizer with newly generated tokens from mbart model.

    Parameters
    ----------
    teacher : str
        The name of mbart model to tokenize sentences
        valied model for MBart50TokenizerFast

    student : str
        The name of marian model to be updated
        valied model for MarianTokenizer

    senetences: List
        The list of sentences to be tokenized by teacher

    CREATE_TOKENS_FILE : bool, Optional, default to False
        Indicate if the text file containing newly added tokens is to be generated.

    TOKENS_FILE_NAME: , Optional, Default to 'Tokens.txt'
        path for Tokens file.


    Returns
    ------
    Updated MarianTokenizer tokenizer
    """

    marian_tokenizer = MarianTokenizer.from_pretrained(student)
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained(teacher)

    sents = sentences

    tokensZoo = []

    for s in tqdm(sents):
        try:
            tokens = mbart_tokenizer.tokenize(s)
            tokens = [a.lstrip("▁") for a in tokens]
            tokensZoo.extend(tokens)
        except:
            print(s, type(s), tokens)

    # discard the duplicates but keep the order
    unique_tokens_zoo = list(dict.fromkeys(tokensZoo))
    if "" in unique_tokens_zoo:
        unique_tokens_zoo.remove("")

    print(f"{len(unique_tokens_zoo)} tokens to be added.")
    print(f"initial vocab size: {len(marian_tokenizer)}")
    marian_tokenizer.add_tokens(list(unique_tokens_zoo), special_tokens=True)
    print(f"final vocab size: {len(marian_tokenizer)}")

    if CREATE_TOKENS_FILE:
        with open(TOKENS_FILE_NAME, "w") as outfile:
            outfile.write(
                "\n".join(list(marian_tokenizer.added_tokens_encoder.keys())))

    return marian_tokenizer


if __name__ == '__main__':
    alt_ds = load_dataset('DeskDown/ALTDataset')
    teacher = "facebook/mbart-large-50-one-to-many-mmt"
    student = 'Helsinki-NLP/opus-mt-en-zh'
    sents = alt_ds["train"]["Sent_yy"]
    marian_tokenizer = update_tokenizer(student=student,
                                        teacher=teacher,
                                        sentences=sents,
                                        CREATE_TOKENS_FILE=True)
