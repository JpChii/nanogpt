import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences to process in parallel
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # lowered lr for deeper layer
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384 # 384 / 6(n_head) --> 64 dimensional heads
n_layer = 6
n_head = 6
dropout = 0.2

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# collecting vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
# encoder: take a string, output sequence of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers and return a string
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(text))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # gernerate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
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

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, self.head_size, bias=False)
        self.query = nn.Linear(n_embd, self.head_size, bias=False)
        self.value = nn.Linear(n_embd, self.head_size, bias=False)
        # Tril is not a model paramters, so with pytroch we've to register is as buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores (affiinites)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        # Perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """
    multiple heads of self attention in parallel, Communication layer
    """

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate heads results along channel dimension
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """
    Computation layer after communication betwee tokens in attention layers
    """
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4), # embd * embd matrix, * 4 from attention is all you need paper to increase computation
            nn.ReLU(), # Non linear activation
            nn.Linear(4 * n_embd, n_embd), # Projection layer
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block communication followed by compuation
    """

    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Using pre normalization deviation from attention is all you need papaer
        x = x + self.sa(self.ln1(x)) # residual connection + computation
        x = x + self.ffwd(self.ln2(x)) # residual connection + computation
        return x

class BigramLanguageModel(nn.Module):
    # Removing vocab_size from constructor as it's a global variable
    def __init__(self):
        super().__init__()
        # For each token lookup nexr character from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Replacing sa and ffd with communication and computaional blocls
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B, T, C), c-> n_embd
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_embd # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to block_size tokens to accomodate position table
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)  # calls forward
            # focus only on last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1)  # -1 is C
            # Get 1 sample from distirbution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled idx to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def train():
    model = BigramLanguageModel()
    m = model.to(device)

    # Create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evalute the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
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

def generate(context, max_new_tokens):
    model = BigramLanguageModel()
    model.load_state_dict(torch.load("nanogpt.pt", map_location=device))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# xb, yb = get_batch("train")
# print(xb.shape)
# Results from colab
# --------------------------

# Step 0: train loss 4.2846, val loss 4.2820
# Step 500: train loss 1.8865, val loss 2.0023
# Step 1000: train loss 1.5361, val loss 1.7221
# Step 1500: train loss 1.3948, val loss 1.6038
# Step 2000: train loss 1.3077, val loss 1.5490
# Step 2500: train loss 1.2523, val loss 1.5153
# Step 3000: train loss 1.2010, val loss 1.4894
# Step 3500: train loss 1.1587, val loss 1.4800
# Step 4000: train loss 1.1222, val loss 1.4800
# Step 4500: train loss 1.0853, val loss 1.4736

# But with price of a breast sast-creatories?
# For my Capitol, the Lude haste of Green:
# The story king o'er match'd the night, his bosom mind;
# But every more like his desired or her,
# And swore let the heirs of the time
# He right; see the duke our dry clooks when he flesh:
# We say, let the waking of warth, with war our drive;
# Which is the old field spice, if waxed the park,
# Frowardly gladling back'd with Clarence, wife comely helps,
# Strucking in the heather cell his own his royal frience.

# HENRY BOLIN
# --------------------------
