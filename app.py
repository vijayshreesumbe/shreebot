import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to generate text using GPT-2
def generate_text(prompt, max_length=100):
    model_name = "gpt2"  # You can use other models like 'gpt2-medium', 'gpt2-large', etc.
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app layout
st.title("Shree's Story Bot ")
st.write("Enter a prompt to generate story:")

# User input for the prompt
user_input = st.text_input("Prompt", "Once upon a time")

# Button to generate text
if st.button("Generate"):
    if user_input:
        generated_text = generate_text(user_input)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt.")
