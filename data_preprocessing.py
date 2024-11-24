import numpy as np
import pandas as pd
import re

def tokenize_text(text : str) -> list[str]:
    "Tokenize the input text into a list of tokens."
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

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