# NMT

## Tokens.txt
Pretrained Marian tokenized Does not support all the languages present in ALT dataset.  
To overcome this limitation, we used mbart50 tokenizer to tokenize asian languages it supports.  
Tokens.txt contains all the tokens that were added to the vocabulary of Marian Tokenizer.  
The list of languages from ALT dataset used to create these tokens is as following:  

* 'fil'
* 'vi'
* 'id'
* 'ms'
* 'ja'
* 'khm'
* 'th'
* 'hi'
* 'my'
* 'zh'

Although chinese is supported by Marian Tokenizer but we included it in the list to have additional tokens which might not have appeared during pre-training of Marian model.
