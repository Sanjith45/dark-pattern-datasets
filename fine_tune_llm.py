from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

# Load train, test, and validation datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("validation.csv")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

# Tokenize the input texts
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True)
# Define training arguments
# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',  # Specify the output directory
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5
)


# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings
)

# Fine-tune the model
trainer.train()

# Make predictions on the test set
predictions = trainer.predict(test_encodings)
pred_labels = predictions.predictions.argmax(axis=1)

# Print classification report
print(classification_report(test_df['target'], pred_labels))

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
