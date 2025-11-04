# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
# ---


# %%capture
# this program is a test of tiktoken, and its ability to separate paragraphs into colored tokens
# based on a selected LLM

import tiktoken

from rich.console import Console

from rich.text import Text

tokenizer = tiktoken.get_encoding("cl100k_base")

console = console()

console = Console()


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = f.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]


clear


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = f.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = f.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]
    for i, token in enumerate(token_strings):
        # simply cylce through the colors
        color = colors[i % len(colors)]
        text_obj.append(token, syle=color)
    # should print in color


tokenize_and_print("hamlet.txt")


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = file.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]
    for i, token in enumerate(token_strings):
        # simply cylce through the colors
        color = colors[i % len(colors)]
        text_obj.append(token, syle=color)
    # should print in color
    console.print(text_obj)


tokenize_and_print("hamlet.txt")


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = file.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]
    for i, token in enumerate(token_strings):
        # simply cylce through the colors
        color = colors[i % len(colors)]
        text_obj.append(token, style=color)
    # should print in color
    console.print(text_obj)


tokenize_and_print("hamlet.txt")


# %notebook token_colorization.ipynb


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = file.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]
    for i, token in enumerate(token_strings):
        # replace space with a middle dot
        token = token.replace(" ", ".")
        # simply cylce through the colors
        color = colors[i % len(colors)]
        text_obj.append(token, style=color)
    # should print in color
    console.print(text_obj)


tokenize_and_print("hamlet.txt")


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = file.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]
    for i, token in enumerate(token_strings):
        # replace space with a middle dot
        token = token.replace(" ", "·")
        # simply cylce through the colors
        color = colors[i % len(colors)]
        text_obj.append(token, style=color)
    # should print in color
    console.print(text_obj)


tokenize_and_print("hamlet.txt")

# %notebook token_colorization.ipynb


def tokenize_and_print(filename):
    with open("hamlet.txt", "r") as file:
        text = file.read()
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    text_obj = Text()
    colors = ["cyan", "magenta", "yellow", "green", "red" "blue"]
    for i, token in enumerate(token_strings):
        # replace space with a middle dot
        token = token.replace(" ", "·")
        # simply cylce through the colors
        color = colors[i % len(colors)]
        text_obj.append(token, style=color)
    # should print in color
    console.print(text_obj)
    # log number of tokens
    print(f"num tokens: {len(tokens)}")


tokenize_and_print("hamlet.txt")
