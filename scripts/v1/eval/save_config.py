import json
path = 'emma-7b/config.json'
# Step 1: Open and load the existing JSON data
with open(path, 'r') as file:
    data = json.load(file)  # Load the JSON data into a dictionary

# Step 2: Append new key-value pairs

data['encoder_version'] = 'v1' 
data['num_learnable_tokens'] = 0 
data['mm_text_select_layer']= -2 
data['mm_text_select_feature'] = 'cls_patch'

# Step 3: Write the updated data back to the JSON file
with open(path, 'w') as file:
    json.dump(data, file, indent=4)
