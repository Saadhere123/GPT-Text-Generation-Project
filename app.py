import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="GPT Text Generator",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 GPT Text Generation App")

generator = pipeline(
    "text-generation",
    model="gpt2"
)

prompt = st.text_area(
    "Enter Prompt",
    height=150
)

max_length = st.slider(
    "Max Length",
    50,
    300,
    150
)

temperature = st.slider(
    "Temperature",
    0.1,
    1.5,
    0.8
)

if st.button("Generate Text"):

    if prompt:

        with st.spinner("Generating..."):

            output = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )

            generated_text = output[0]["generated_text"]

            st.subheader("Generated Output")

            st.write(generated_text)
