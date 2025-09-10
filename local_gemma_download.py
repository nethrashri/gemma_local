from huggingface_hub import login
login(token="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")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model
model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"}
)

# Define the chat messages
messages = [
    {"role": "user", "content": "What is the largest country in the world by land area?"}
]

# Apply the chat template and run inference
outputs = pipe(
    messages,
    max_new_tokens=256,
    disable_compile=True
)

# Print the generated text
print(outputs[0]['generated_text'])