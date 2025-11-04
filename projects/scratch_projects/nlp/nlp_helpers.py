def test_corpus_l() -> str:
    with open("macbeth.txt", "r") as file:
        macbeth = file.read()
    return macbeth

def test_corpus_s() -> str:
    with open("hamlet.txt", "r") as file:
        macbeth = file.read()
    return macbeth

from collections import Counter
def vocab(text, split_char):
    words = text.split(split_char)
    counts = Counter(words)
    return counts

import re
def removeNonAlphaNumeric(s: str) -> str:
    # Split the string on "."
    sentences = s.split(".")
    # Remove non-alphanumeric characters from each sentence
    # This regex should remove non alphanum but leave in the punctuation, excluding commas
    cleaned_sentences = [re.sub(r'[^a-zA-Z0-9,\'\.\?! \n]', ' ', sentence) for sentence in sentences]
    # Rejoin the sentences with "."
    result_string = ".".join(cleaned_sentences)
    return result_string