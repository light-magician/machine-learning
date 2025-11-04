# coding: utf-8
with open("hamlet.txt", "r") as hamlet:
    text = file.read()
    
with open("hamlet.txt", "r") as hamlet:
    text = hamlet
    .read()
with open("hamlet.txt", "r") as hamlet:
    text = hamlet.read()
    
print(text)
# lets read through the file and map every pair of two letters
get_ipython().run_line_magic('clear', '')
sentences = text.split("\n")
counter = Counter()
import Counter
help(Counter)
import clear
get_ipython().run_line_magic('clear', '')
from colletions import Counter
from collections import Counter
get_ipython().run_line_magic('clear', '')
counter = Counter()
type(sentences)
words = list()
for sentence in sentences:
    word_splt = sentence.split(" ")
    words.add(word_splt)
    
for sentence in sentences:
    word_splt = sentence.split(" ")
    words.append(word_splt)
    
words[:20]
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', 'words = list()\nfor sentence in sentences:\n    splt = sentence.split(" ")\n    for wrd in splt: \n        words.append(wrd)\n\n        \n')
words[:20]
get_ipython().run_cell_magic('capture', '', '# go through the words and get the bigrams,\n# first clean the data\nfor word in words:\n    # remove comma, apostraphy, and caps\n    word = string.lower()\n    \n')
# I don't understand zip so I cant do this pythonically
