# NanoGpt

Goal: How GPT(Generative Pretrained Transformers) works under the hood.

Source: [Karpathy's NanoGpt](https://github.com/karpathy/nanoGPT)

## Process

We're gonna write *transformers*(core of GPT) and train on a small dataset - `tiny shakespeare` to predict sequence of characters given the context.

At end the code can optionally load GPT2 weights from OpenAI

## Streamlit application

File used for streamlit app

* app.py --> streamlit app itself
* vocab.json --> loaded itos to this file to use it as prompt to generate text
