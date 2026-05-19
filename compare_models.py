from transformers import pipeline

gpt2_generator = pipeline(
    "text-generation",
    model="gpt2"
)

neo_generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M"
)

prompt = "Artificial Intelligence in healthcare"

print("\nGPT-2 OUTPUT:\n")

gpt2_output = gpt2_generator(
    prompt,
    max_length=100
)

print(gpt2_output[0]["generated_text"])

print("\n" + "="*80)

print("\nGPT-Neo OUTPUT:\n")

neo_output = neo_generator(
    prompt,
    max_length=100
)

print(neo_output[0]["generated_text"])
