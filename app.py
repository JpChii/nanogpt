import streamlit as st
from bigram_v2 import BigramLanguageModel
import torch
import json

st.title("NanoGpt to generate shakespearian text")

st.warning("This demo uses a genrative model and will take some time to generate text on free resources...Please wait!",
           icon="⚠️")

# Load the model
device = "cuda" if torch.cuda.is_available() else"cpu"
model_name = "nanogpt"
model = BigramLanguageModel()
model.load_state_dict(
    torch.load(f"{model_name}.pt",map_location=device)
)

def decode_generated_text(output):
    decoder = lambda l: "".join([itos[i.item()] for i in l.flatten()])
    decoded_output = ""
    try:
        decoded_output = decoder(output)
    except KeyError as e:
        key = str(e).strip("'")
        print(f"KeyError was thrown for key: {key}")
    return decoded_output

# Loading the required to generate text
with open("vocab.json") as file:
    contexts = json.load(file)
itos = {i: ch for ch, i in contexts.items()}

# Dropdown default option
default_option = "Select option"

# Demo parameters
# Max sequence length for genretive text
max_sequence_length_options = ["500", "1000", "1500", "2000", "2500"]
# Info on Demo
st.sidebar.title("Info & Params")
st.sidebar.write(
    """
    This demo is built on top of a character level language model from shakespere text dataset. This is built by referring to andrei karpathy's nanogpt.

    Demo has two options:
    * Max Length - Number of characters to generate
    * Initial context - This is the initial character context to generate text, simply put starting point of text generation.
    """ 
)
st.sidebar.markdown("[Source Code](https://github.com/JpChii/nanogpt)")
selected_seq_length = st.sidebar.selectbox(
    "Max Length",
    max_sequence_length_options,
)

# Select an option from the loaded vocab
selected_option = st.sidebar.selectbox(
    "Inital context",
    list(contexts.keys())[:2] + list(contexts.keys())[15:],
    index=list(contexts.keys()).index(default_option), # Default option on dropdown
)

if selected_option != default_option:

    selected_value = contexts[selected_option]

    st.empty()
    st.write("Generating text...")
    if selected_option == "Default option":
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.tensor([[selected_value]], dtype=torch.long, device=device)
    generated_output = model.generate(
        idx=context,
        max_new_tokens=int(selected_seq_length)
    )
    st.write("Here you go...")
    st.markdown(
    decode_generated_text(
        output=generated_output,
    ),
    unsafe_allow_html=True
    )