import json
import re
import random

def clean_text(text):
    """
    Cleans raw text using explicit loops (no list comprehensions).
    """
    #create only printable text string
    printable_text = ""
    for char in text:
        if char.isprintable():
            printable_text += char
    
    # 2. Normalize whitespace
    words = printable_text.split()
    
    cleaned_text = ""
    for i in range(len(words)):
        cleaned_text += words[i]
        if i < len(words) - 1:
            cleaned_text += " "
            
    return cleaned_text

def load_and_clean_data(file_path):
    """
    Reads JSONL, cleans the 'text' field, and returns a list.
    """
    cleaned_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != "":
                    # Parse JSON
                    data_dict = json.loads(line)
                    raw_text = data_dict.get("text", "")
                    
                    # Clean text
                    cleaned = clean_text(raw_text)
                    
                    # Only keep if not empty
                    if cleaned != "":
                        cleaned_list.append(cleaned)
    except FileNotFoundError:
        print("Error: Could not find " + file_path)
        
    return cleaned_list

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Partitions data into three sets using manual indexing.
    """
    # Shuffle for randomness
    random.seed(2023111005)
    random.shuffle(data)
    
    total_count = len(data)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)
    
    # Slicing is standard Python; we extract the portions
    train_set = data[0 : train_end]
    val_set = data[train_end : val_end]
    test_set = data[val_end : ]
    
    return train_set, val_set, test_set

def save_list_to_file(data_list, filename):
    """
    Saves each cleaned sentence to a new line in a text file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(item + "\n")

def create_subset(source_file, train_output, test_output, train_limit, test_limit):
    """
    Reads from a large source file and saves specific amounts to new files.
    """
    print("Processing: " + source_file)
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f_in:
            # 1. Create the small training sample
            with open(train_output, 'w', encoding='utf-8') as f_train:
                for i in range(train_limit):
                    line = f_in.readline()
                    if not line:
                        break
                    f_train.write(line)
            
            # 2. Create the small testing sample (the next 10,000 lines)
            with open(test_output, 'w', encoding='utf-8') as f_test:
                for i in range(test_limit):
                    line = f_in.readline()
                    if not line:
                        break
                    f_test.write(line)
                    
        print("Done. Created " + train_output + " and " + test_output)
        
    except FileNotFoundError:
        print("Error: Could not find " + source_file + ". Make sure Task 1.1 was run first.")

#run

# 1. Process English
print("Loading and cleaning English corpus...")
en_data = load_and_clean_data("./data/datasets/cc100_en.jsonl")
en_train, en_val, en_test = split_data(en_data)

# 2. Process Mongolian
print("Loading and cleaning Mongolian corpus...")
mn_data = load_and_clean_data("./data/datasets/cc100_mn.jsonl")
mn_train, mn_val, mn_test = split_data(mn_data)

# Print Summary Stats
print("\n" + "="*30)
print("PARTITION SUMMARY")
print("="*30)
print("English   - Train:", len(en_train), "| Val:", len(en_val), "| Test:", len(en_test))
print("Mongolian - Train:", len(mn_train), "| Val:", len(mn_val), "| Test:", len(mn_test))

# --- PREVIEW TOP 100 SAMPLES ---

print("\n" + "="*30)
print("PREVIEW: Top 100 English Training Samples")
print("="*30)
for i in range(min(100, len(en_train))):
    print(f"{i+1}: {en_train[i]}")

print("\n" + "="*30)
print("PREVIEW: Top 100 Mongolian Training Samples")
print("="*30)
for i in range(min(100, len(mn_train))):
    print(f"{i+1}: {mn_train[i]}")


print("\nSaving cleaned files...")
save_list_to_file(en_train, "./data/datasets/en_final_train.txt")
save_list_to_file(en_test, "./data/datasets/en_final_test.txt")
save_list_to_file(mn_train, "./data/datasets/mn_final_train.txt")
save_list_to_file(mn_test, "./data/datasets/mn_final_test.txt")
print("Done!")

TRAIN_SIZE = 100000
TEST_SIZE = 50000

print("\n" + "="*30)
print("CREATING SUBSETS")
print("="*30)

# Process English
create_subset(
    "./data/datasets/en_final_train.txt", 
    "./data/datasets/en_sample_train.txt", 
    "./data/datasets/en_sample_test.txt", 
    TRAIN_SIZE, 
    TEST_SIZE
)

# Process Mongolian
create_subset(
    "./data/datasets/mn_final_train.txt", 
    "./data/datasets/mn_sample_train.txt", 
    "./data/datasets/mn_sample_test.txt", 
    TRAIN_SIZE, 
    TEST_SIZE
)