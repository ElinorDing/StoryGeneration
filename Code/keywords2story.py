from transformers import AutoTokenizer, T5ForConditionalGeneration,T5Model,Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
import evaluate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "/Users/vonnet/Master/masterProject/Dataset/FinalDataset/dataset.csv"
checkpoint = "t5-base"
output_dir = '/Users/vonnet/Master/masterProject/Code/output_dir'

read_dataset = pd.read_csv(data_path)

train_dataset = read_dataset.sample(frac=0.8, random_state=42)
test_dataset = read_dataset.drop(train_dataset.index)

# Convert training and testing dataset from pandas dataframe to HuggingFace Dataset format
hg_train_dataset = Dataset.from_pandas(train_dataset)
hg_test_dataset = Dataset.from_pandas(test_dataset)

# a tokenizer convert text into numbers to use as model input. Each number represents a token, which can be a word, punctuation, or special tokens.
# The pretrained model determine how the text is tokenized
# AutoTokenizer.from_pretrained() will automatically download the vocabulary from the pretrained model hub if it has not been downloaded before

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
task_prefix = "Create a Social Story about: "
# Function to tokenize the training and testing dataset
def tokenize_function(data):
    model_input = tokenizer(data["keyword"], max_length=5, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(data["stories"], max_length=300, padding="max_length", truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    
    model_input["labels"] = labels
    return model_input

# # Tokenize the training and testing dataset
tokenized_train_dataset = hg_train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = hg_test_dataset.map(tokenize_function, batched=True)


# Load the pretrained model
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)  # GPT2
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device) # T5 with language modeling head
# model = T5Model.from_pretrained(checkpoint).to(device) # T5 without any specific head on top 

training_args = TrainingArguments(
    output_dir=output_dir, #The output directory
    logging_dir=output_dir, #Logs
    logging_strategy='epoch',
    logging_steps=10,
    num_train_epochs=10, #Total number of training epochs to perform
    per_device_train_batch_size=2, #Batch size per device during training
    per_device_eval_batch_size=2, #Batch size for evaluation
    do_predict=True,
    learning_rate=5e-5,
    seed=42,
    save_strategy='epoch',
    save_steps=10,
    evaluation_strategy='epoch',
    eval_steps=10,
    load_best_model_at_end=True
)

# # set evaluate metric as HuggingFace Trainer does not evaluate the model automatically during training
# # Which evaluation metric to use for our task
# # Accuracy can be used when dataset are balanced

# Load a metric from the Hugging Face datasets library
metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: v for k, v in result.items()}

    # predictions = np.argmax(predictions, axis=1)
    # return metric.compute(predictions=predictions, references=labels)

# # Train the model using the transformer Trainer class
# # Model is the model for training, evaluation or prediction by the Trainer
# # args is TrainingArguments to twist the Trainer
# # train_dataset is the training dataset
# # eval_dataset is the evaluation dataset
# # compute_metrics is the function to compute metrics during evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics
)

# # Train the model
trainer.train()

y_pred = trainer.predict(tokenized_test_dataset)