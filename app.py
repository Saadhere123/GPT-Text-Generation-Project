import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

st.set_page_config(
    page_title="GPT Text Generator",
    page_icon="🤖"
)

st.title("🤖 GPT Text Generation")

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator

generator = load_model()

prompt = st.text_area("Enter Prompt")

if st.button("Generate"):

    if prompt:

        output = generator(
            prompt,
            max_length=120,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

        st.write(output[0]["generated_text"])
