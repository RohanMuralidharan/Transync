"""
Example: Fine-tune mbart on your own translation data
This creates a NEW model that's uniquely yours
"""
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# 1. Load the base model
model_name = "facebook/mbart-large-50-one-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50Tokenizer.from_pretrained(model_name)

# 2. Prepare your training data
# Format: list of (source_text, target_text) pairs
your_data = [
    ("Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
    ("Good morning", "सुप्रभात"),
    # Add your own data here - at least 1000+ pairs for good results
]

# Convert to HuggingFace Dataset format
dataset = Dataset.from_dict({
    "source": [x[0] for x in your_data],
    "target": [x[1] for x in your_data]
})

# 3. Tokenize
def preprocess(examples):
    inputs = tokenizer(examples["source"], max_length=128, truncation=True)
    targets = tokenizer(examples["target"], max_length=128, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# 4. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./my_finetuned_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=100,
    fp16=torch.cuda.is_available(),
)

# 5. Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()

# 6. Save your NEW model
model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

print("Your custom model is ready!")