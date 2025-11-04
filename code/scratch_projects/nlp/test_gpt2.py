# coding: utf-8

# the following was an ipython session
from transformers import pipeline, set_seed

set_seed(42)
generator = pipeline("text-generation", model="gpt2")
generator("hello I am a language model,", max_length=30, num_return_sequences=5)
baby_function_call = "the following word that is most like 'library' from the list [book, dragon, red, was, something] is"
generator(baby_function_call, max_length=2, num_return_sequences=5)
say_this = 'say one of the following words ["book", "dog", "cat", "red", "ocean"]'
generator(say_this, max_length=1, num_return_sequences=5)
# gpt 2 small can't do this at all
