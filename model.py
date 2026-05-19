from transformers import pipeline
import os

os.makedirs("outputs", exist_ok=True)

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_text(prompt):

    output = generator(
        prompt,
        max_length=150,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1
    )

    generated_text = output[0]["generated_text"]

    with open("outputs/generated_output.txt", "a", encoding="utf-8") as f:
        f.write(f"\nPROMPT:\n{prompt}\n")
        f.write(f"\nGENERATED TEXT:\n{generated_text}\n")
        f.write("\n" + "="*80 + "\n")

    return generated_text

if __name__ == "__main__":

    prompt = input("Enter your prompt: ")

    result = generate_text(prompt)

    print("\nGenerated Text:\n")
    print(result)
