import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification, 
                          TrainingArguments, Trainer, EarlyStoppingCallback)
import torch.nn as nn
import torch
import numpy as np
import re

# 1. Load your dataset (assuming CSV with columns "headline" and "is_sarcastic")
data = pd.read_csv("train.csv")

# 1a. Convert any dictionary entries in the "headline" column to strings.
def extract_text(x):
    if isinstance(x, dict):
        return x.get("content", str(x))
    return str(x)

data["headline"] = data["headline"].apply(extract_text)

# 1b. Clean the text in the "headline" column
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data["headline"] = data["headline"].apply(clean_text)

# 2. Split the DataFrame into train and validation (90:10 ratio)
train_df, val_df = train_test_split(data, test_size=0.10, random_state=42, stratify=data["is_sarcastic"])

# 3. Convert Pandas DataFrames into Hugging Face Datasets using Dataset.from_pandas()
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Create a DatasetDict for convenience
datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# 4. Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 5. Tokenize the datasets; note that we use "headline" instead of "text"
def tokenize_function(examples):
    return tokenizer(examples["headline"], padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Rename "is_sarcastic" to "labels" so the Trainer can find them.
tokenized_datasets = tokenized_datasets.rename_column("is_sarcastic", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 6. Load the DistilBERT model. The number of labels is determined by the number of unique values in "is_sarcastic".
base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=len(data["is_sarcastic"].unique())
)

# 7. Freeze all DistilBERT parameters except the last transformer layer.
for param in base_model.distilbert.parameters():
    param.requires_grad = False
for param in base_model.distilbert.transformer.layer[-1].parameters():
    param.requires_grad = True

# 8. Create a custom model wrapper to add dropout.
class CustomDistilBert(nn.Module):
    def __init__(self, base_model, dropout_rate=0.3):
        super(CustomDistilBert, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        # Use the classifier from base_model.
        self.classifier = base_model.classifier

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model.distilbert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]  # Use CLS token representation
        dropped = self.dropout(hidden_state)
        logits = self.classifier(dropped)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}

# Replace base_model with the custom model
model = CustomDistilBert(base_model)

# 9. Define training arguments with weight decay and early stopping.
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)

# 10. Define a simple compute_metrics function (calculating accuracy)
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# 11. Create the Trainer and add EarlyStoppingCallback.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 12. Train the model.
trainer.train()

