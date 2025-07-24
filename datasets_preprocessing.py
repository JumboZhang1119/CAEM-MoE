import warnings
import torch
import os
from transformers import T5TokenizerFast
from datasets import load_dataset, DatasetDict, load_from_disk

# --- Environment Configuration ---
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.set_device(0)

# --- Define storage paths ---
AG_NEWS_PATH = "/path/to/your/storage/ag_news_dataset"
SQUAD_PATH = "/path/to/your/storage/squad_dataset"
XSUM_PATH = "/path/to/your/storage/xsum_dataset"


# --- Tokenizer Initialization ---
tokenizer = T5TokenizerFast.from_pretrained(f"google/switch-base-8")

# --- Global Configurations ---
config = {
    'max_input_length': 1000,
    'max_decoder_length': 200,
}

# --- AG News Dataset ---
def tokenize_ag_news(example):
    inputs = tokenizer(
        "Classify: " + example['text'],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_input_length'],
        return_attention_mask=True
    )
    label_map = {
        0: "World", 1: "Sports", 2: "Business", 3: "Science and Technology"
    }
    label_text = label_map[example['label']]
    label_tokens = tokenizer(
        label_text,
        padding='max_length', 
        truncation=True, 
        max_length=config['max_decoder_length']
    )
    labels_input_ids = torch.tensor(label_tokens['input_ids']).squeeze()
    labels_input_ids[labels_input_ids == 0] = -100 # Use pad_token_id
    return {
        'input_ids': torch.tensor(inputs['input_ids']).squeeze(),
        'attention_mask': torch.tensor(inputs['attention_mask']).squeeze(),
        'labels': labels_input_ids,
    }
    
# --- SQuAD Dataset ---
def tokenize_squad_train(example): # For single answer
    inputs = tokenizer(
        "Question: " + example['question'] + " Context: " + example['context'],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_input_length'],
        return_attention_mask=True
    )
    label_tokens = tokenizer(
        example['answers']['text'],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_decoder_length']
    )
    labels_input_ids = torch.tensor(label_tokens['input_ids']).squeeze()
    labels_input_ids[labels_input_ids == 0] = -100
    return {
        'input_ids': torch.tensor(inputs['input_ids']).squeeze(),
        'attention_mask': torch.tensor(inputs['attention_mask']).squeeze(),
        'labels': labels_input_ids,
    }
def tokenize_squad_val(example): # For multiple answers
    inputs = tokenizer(
        "Question: " + example['question'] + " Context: " + example['context'],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_input_length'],
        return_attention_mask=True
    )
    answers = example['answers']['text']
    # Select the longest answer as the ground truth
    longest_idx = max(range(len(answers)), key=lambda i: len(answers[i]))
    
    label_tokens = tokenizer(
        answers[longest_idx],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_decoder_length']
    )
    labels_input_ids = torch.tensor(label_tokens['input_ids']).squeeze()
    labels_input_ids[labels_input_ids == 0] = -100
    return {
        'input_ids': torch.tensor(inputs['input_ids']).squeeze(),
        'attention_mask': torch.tensor(inputs['attention_mask']).squeeze(),
        'labels': labels_input_ids,
    }

# --- XSum Dataset ---
def tokenize_xsum(example):
    inputs = tokenizer(
        "Summarize: " + example['document'],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_input_length'],
        return_attention_mask=True
    )
    label_tokens = tokenizer(
        example['summary'],
        padding='max_length', 
        truncation=True, 
        max_length=config['max_decoder_length']
    )
    labels_input_ids = torch.tensor(label_tokens['input_ids']).squeeze()
    labels_input_ids[labels_input_ids == 0] = -100
    return {
        'input_ids': torch.tensor(inputs['input_ids']).squeeze(),
        'attention_mask': torch.tensor(inputs['attention_mask']).squeeze(),
        'labels': labels_input_ids,
    }

# --- AG News Pipeline ---
print("Processing AG News...")
ag_news = load_dataset("ag_news")
ag_news_train_shuffled = ag_news['train'].shuffle(seed=42)
train_validation_split = ag_news_train_shuffled.train_test_split(test_size=0.1, seed=42)
ag_news_dataset = DatasetDict({
    'train': train_validation_split['train'],
    'validation': train_validation_split['test'],
    'test': ag_news['test']
})
tokenized_ag_news = ag_news_dataset.map(tokenize_ag_news, remove_columns=['label', 'text'])

# --- SQuAD Pipeline ---
print("Processing SQuAD...")
squad = load_dataset("squad")
tokenized_squad_train = squad['train'].shuffle(seed=42).map(tokenize_squad_train, remove_columns=['id', 'title', 'question', 'context', 'answers'])
tokenized_squad_test = squad['validation'].map(tokenize_squad_val, remove_columns=['title', 'question', 'context'])
tokenized_squad = DatasetDict({
    'train': tokenized_squad_train,
    'validation': tokenized_squad_test,
    'test': tokenized_squad_test
})

# --- XSum Pipeline ---
print("Processing XSum...")
xsum = load_dataset("xsum", trust_remote_code=True)
xsum_train = xsum['train'].shuffle(seed=42)
xsum_validation = xsum['validation']
xsum_test = xsum['test']
xsum_dataset = DatasetDict({
    'train': xsum_train,
    'validation': xsum_validation,
    'test': xsum_test
})
tokenized_xsum = xsum_dataset.map(tokenize_xsum, remove_columns=['document', 'summary', 'id'])

# --- Saving to disk ---
print("Saving processed datasets to disk...")
tokenized_ag_news.save_to_disk(AG_NEWS_PATH)
tokenized_squad.save_to_disk(SQUAD_PATH)
tokenized_xsum.save_to_disk(XSUM_PATH)

# --- Loading from disk ---
print("Loading processed datasets from disk...")
tokenized_ag_news = load_from_disk(AG_NEWS_PATH)
tokenized_squad = load_from_disk(SQUAD_PATH)
tokenized_xsum = load_from_disk(XSUM_PATH)

# --- Verification ---
print("\nFinal Dataset Structures:")
print(tokenized_ag_news)
print(tokenized_squad)
print(tokenized_xsum)