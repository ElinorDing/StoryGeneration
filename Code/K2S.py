from transformers import AutoTokenizer, T5ForConditionalGeneration,DataCollatorWithPadding,T5Model,Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

data_path = "/Users/vonnet/Master/StoryGeneration/FinalDataset/dataset.csv"
checkpoint = "t5-base"
output_dir = '/Users/vonnet/Master/StoryGeneration/Code/output_dir'

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
    input_with_prefix = [task_prefix + inp for inp in data["keywords"]]
    model_input = tokenizer(input_with_prefix, max_length=12, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(data["story"], max_length=300, padding="max_length", truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    
    model_input["labels"] = labels
    return model_input

# # Tokenize the training and testing dataset
tokenized_train_dataset = hg_train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = hg_test_dataset.map(tokenize_function, batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["keywords", "story", "__index_level_0__"])
tokenized_train_dataset.set_format("torch")
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["keywords", "story", "__index_level_0__"])
tokenized_test_dataset.set_format("torch")

# Load the pretrained model
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)  # GPT2
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device) # T5 with language modeling head
# model = T5Model.from_pretrained(checkpoint).to(device) # T5 without any specific head on top 



# We create a data_collator that will dynamically pad our inputs
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

epochs = 1
batch_size = 4
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5) # Adam with proper weight decay

# The learning rate scheduler will decrease the learning rate as the number of epoch increase
# Progressively decay learning rate to zero 
train_steps = len(tokenized_train_dataset) * epochs // batch_size
warmup_steps = train_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)

train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

progress_bar_train = tqdm(range(train_steps))
progress_bar_test = tqdm(range(epochs * len(test_dataloader) ))
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward() #  compute gradient
        optimizer.step() # update weights to reduce loss
        scheduler.step() # decay learning rate
        optimizer.zero_grad()
        progress_bar_train.update(1)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # human_readable_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            output_id = model.generate(input_ids, attention_mask=attention_mask, max_length=300)
            output = tokenizer.batch_decode(output_id, skip_special_tokens=True)
            progress_bar_test.update(1)
            

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