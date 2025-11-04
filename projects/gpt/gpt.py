import pickle

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
# Reduced footprint so CPU/Metal training completes in reasonable time.
batch_size = 16  # smaller batch keeps per-step cost low on CPU
block_size = 128  # shorter context lowers attention memory footprint
max_iters = 3000  # fewer steps to shorten overall training time
eval_interval = 500
learning_rate = 3e-4
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
eval_iters = 200
n_embd = 128  # smaller embedding makes each projection cheaper
n_head = 4  # fewer attention heads to cut compute load
n_layer = 4  # shallower stack keeps forward/backward passes fast
dropout = 0.2
# ------------

MODEL_PKL_PATH = "gpt_model.pkl"
TOKENIZER_TYPE = "tiktoken"  # set to "char" for simple character-level tokens
TIKTOKEN_ENCODING = "r50k_base"  # GPT-3/GPT-2 tokenizer


def build_char_tokenizer(corpus):
    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(text):
        return [stoi[c] for c in text]

    def decode(token_ids):
        return "".join(itos[i] for i in token_ids)

    return {
        "encode": encode,
        "decode": decode,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(chars),
        "type": "char",
    }


def build_tiktoken_tokenizer(encoding_name=TIKTOKEN_ENCODING):
    if tiktoken is None:
        raise ImportError(
            "tiktoken is required for tokenizer_type='tiktoken'. Install it via pip."
        )
    encoding = tiktoken.get_encoding(encoding_name)

    def encode(text):
        return encoding.encode(text)

    def decode(token_ids):
        return encoding.decode(token_ids)

    return {
        "encode": encode,
        "decode": decode,
        "stoi": None,
        "itos": None,
        "vocab_size": encoding.n_vocab,
        "type": "tiktoken",
        "encoding_name": encoding_name,
    }


def initialize_tokenizer(corpus, tokenizer_type=TOKENIZER_TYPE):
    if tokenizer_type == "tiktoken":
        return build_tiktoken_tokenizer()
    if tokenizer_type == "char":
        return build_char_tokenizer(corpus)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer_config = initialize_tokenizer(text, tokenizer_type=TOKENIZER_TYPE)
encode = tokenizer_config["encode"]
decode = tokenizer_config["decode"]
vocab_size = tokenizer_config["vocab_size"]
stoi = tokenizer_config.get("stoi")
itos = tokenizer_config.get("itos")

encoded_ids = encode(text)
num_tokens = len(encoded_ids)
tokenizer_label = tokenizer_config.get("encoding_name", tokenizer_config.get("type"))
print(
    f"Tokenizer: {tokenizer_config['type']} ({tokenizer_label}), vocab size {vocab_size:,}, "
    f"corpus has {len(text):,} characters → {num_tokens:,} tokens"
)

# Train and test splits
data = torch.tensor(encoded_ids, dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


"""
Attention:
    Mechanism that lets a model dynamically
    focus on the most relevant parts of an input sequence 
    when producing an output by computing weighted 
    relationships between all tokens. 

Attention Heads:
    The individual submodules within the attn mechanism. 
    Each head learns to capture different types of 
    relationships, or context shifts, by operating on its 
    own set of query, key, and value projections.
"""


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention scores (affinities)
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def save_model_to_pickle(model, path=MODEL_PKL_PATH, metadata=None):
    """Serialize the Transformer weights and training metadata to a pickle file."""

    original_device = next(model.parameters()).device
    model_cpu = model.to("cpu")
    state_dict = {k: v.cpu() for k, v in model_cpu.state_dict().items()}
    tokenizer_payload = {
        "type": tokenizer_config.get("type"),
        "vocab_size": vocab_size,
    }
    if tokenizer_payload["type"] == "tiktoken":
        tokenizer_payload["encoding_name"] = tokenizer_config.get(
            "encoding_name", TIKTOKEN_ENCODING
        )
    if stoi is not None and itos is not None:
        tokenizer_payload["stoi"] = stoi
        tokenizer_payload["itos"] = itos
    package = {
        "state_dict": state_dict,
        "config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
        },
        "tokenizer": tokenizer_payload,
        "metadata": metadata or {},
    }
    with open(path, "wb") as f:
        pickle.dump(package, f)
    model_cpu.to(original_device)


def load_model_from_pickle(path=MODEL_PKL_PATH, map_location=None):
    """Restore a Transformer model from a pickle produced by save_model_to_pickle."""

    with open(path, "rb") as f:
        package = pickle.load(f)

    config = package.get("config", {})
    expected = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
    mismatched = {
        k: (config.get(k), expected[k])
        for k in expected
        if config.get(k) != expected[k]
    }
    if mismatched:
        raise ValueError(
            f"Saved model config does not match current hyperparameters: {mismatched}"
        )

    restored_model = GPTLanguageModel()
    restored_model.load_state_dict(package["state_dict"])
    target_device = map_location if map_location is not None else device
    restored_model.to(target_device)

    tokenizer_payload = package.get("tokenizer")
    if tokenizer_payload:
        global tokenizer_config, encode, decode, stoi, itos
        tokenizer_type = tokenizer_payload.get("type")
        if tokenizer_type == "char":
            stoi = tokenizer_payload.get("stoi", stoi)
            itos = tokenizer_payload.get("itos", itos)

            def _char_encode(text, mapping=stoi):
                return [mapping[c] for c in text]

            def _char_decode(token_ids, inverse=itos):
                return "".join(inverse[i] for i in token_ids)

            tokenizer_config = {
                "encode": _char_encode,
                "decode": _char_decode,
                "stoi": stoi,
                "itos": itos,
                "vocab_size": tokenizer_payload.get("vocab_size", vocab_size),
                "type": "char",
            }
        elif tokenizer_type == "tiktoken":
            encoding_name = tokenizer_payload.get("encoding_name", TIKTOKEN_ENCODING)
            tokenizer_config = build_tiktoken_tokenizer(encoding_name)
            stoi = tokenizer_config.get("stoi")
            itos = tokenizer_config.get("itos")
        else:
            raise ValueError(
                f"Unsupported tokenizer type in checkpoint: {tokenizer_type}"
            )

        encode = tokenizer_config["encode"]
        decode = tokenizer_config["decode"]

    return restored_model


model = GPTLanguageModel()
m = model.to(device)
if device == "mps":
    print("Using device: metal")
elif device == "cpu":
    print("Using device: cpu")
else:
    print(f"Using device: {device}")
# print the number of parameters in the model
num_params = sum(p.numel() for p in m.parameters())
print(f"Model size: {num_params / 1e6:.2f} M parameters ({num_params:,} total)")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

save_model_to_pickle(model, MODEL_PKL_PATH)
print(f"Saved trained model to {MODEL_PKL_PATH}")
loaded_model = load_model_from_pickle(map_location=device)
print("Reloaded model from pickle and moved to target device")
