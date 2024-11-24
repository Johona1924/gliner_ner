import numpy as np
import pandas as pd

# join tokens in 'tokeinzed_text'
def join_tokens(tokens):
    # code from Gliner_Studio: https://colab.research.google.com/drive/1Kl3TrpiGBpMw569ek_AL6Ee3uqBK-Gfw?usp=sharing
    # Joining tokens with space, but handling special characters correctly
    text = ""
    for token in tokens:
        if token in {",", ".", "!", "?", ":", ";", "..."}:
            text = text.rstrip() + token
        else:
            text += " " + token
    return text.strip()

def gliner_ner_format_text_match(sample):
    tokenized_text = sample['tokenized_text']
    ners = sample['ner']
    ners = sorted(ners,key = lambda x: x[0])

    print("----Matching ner_labels to tokenized_text-----")

    for ner in ners:
        start = ner[0]
        end_token = ner[1]
        print(join_tokens(tokenized_text[start:end_token + 1])," --> ", ner[2],end = '\n')

def tokenize_text(text : str) -> list[str]:
    "Tokenize the input text into a list of tokens."
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


