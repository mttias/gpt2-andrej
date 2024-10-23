import tiktoken

enc = tiktoken.get_encoding("gpt2")

# open data
with open("input.txt", "r") as file:
    raw_text = file.read()

print(raw_text[:1000])

tokens = enc.encode(raw_text)
print(tokens[:100])

import torch

buf = torch.tensor(tokens[: 24 + 1])  # B x T + 1
x = buf[:-1].view(4, 6)  # B x T
y = buf[1:].view(4, 6)  # B x T
print(x)
print(y)
