import h5py
import numpy as np
import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    def __init__(self, captions, tokenizer, max_len=128):
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.encode_caption(self.captions[idx])
        return idx, item['input_ids'].squeeze(), item['attention_mask'].squeeze()

    def __len__(self):
        return len(self.captions)

    def encode_caption(self, caption):
        return self.tokenizer.encode_plus(
            caption,
            max_length=self.max_len,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt')


def get_data_from_h5_bert(file):
    with h5py.File(file, 'r') as ds:
        captions = ds['captions'][:]
    return captions


def remove_punctuation(string):
    return string.replace('.', '').replace(',', '').lower()


def get_embeddings(model, dataloader, device, num_hidden_states=4, operation='sum'):
    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        batch_outputs = []
        hs = [i for i in range(-(num_hidden_states), 0)]
        len_hs = len(hs) * 768 if (operation == 'concat') else 768
        print('(Last) Hidden states to use:', hs, ' -->  Embedding size:', len_hs)

        for idx, input_ids, attention_masks in dataloader:
            input_ids = input_ids.to(device)  # to gpu (if presented)
            attention_masks = attention_masks.to(device)  # to gpu (if presented)
            out = model(input_ids=input_ids, attention_mask=attention_masks)
            hidden_states = out['hidden_states']
            last_hidden = [hidden_states[i] for i in hs]

            if operation == 'sum':
                # stack list of 3D-Tensor into 4D-Tensor
                # 3D [(batch_size, tokens, 768)] -> 4D (hidden_states, batch_size, tokens, 768)
                hiddens = torch.stack(last_hidden)
                # sum along 0th dimension -> 3D (batch_size, tokens, output_dim)
                resulting_states = torch.sum(hiddens, dim=0).squeeze()
            elif operation == 'concat':
                # concat list of 3D-Tensor into 3D-Tensor
                # 3D [(batch_size, tokens, 768)] -> 3D (batch_size, tokens, 768 * list_length)
                resulting_states = torch.cat(tuple(last_hidden), dim=2)
            else:
                raise Exception('unknown operation ' + str(operation))

            # token embeddings to sentence embedding via token embeddings averaging
            # 3D (batch_size, tokens, resulting_states.shape[2]) -> 2D (batch_size, resulting_states.shape[2])
            sentence_emb = torch.mean(resulting_states, dim=1).squeeze()
            batch_outputs.append(sentence_emb)

        # vertical stacking (along 0th dimension)
        # 2D [(batch_size, resulting_states.shape[2])] -> 2D (num_batches * batch_size, resulting_states.shape[2])
        output = torch.vstack(batch_outputs)
        return output.cpu().numpy()  # return to cpu (or do nothing), convert to numpy


def save_embeddings_hdf5(out_file, embeddings):
    with h5py.File(out_file, 'w') as hf:
        print("Saved as '.h5' file to", out_file)
        hf.create_dataset('embeddings', data=embeddings)


########################################################################################################################
# Constants
print('# Constants')
########################################################################################################################

DATASET = 'RSICD'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FILEPATH = r'./data/'
OPERATION = 'concat'  # 'sum' or 'concat'
NUM_HIDDEN = 4  # number of last hidden states to use 1 - 12

print("DATASET:", DATASET)
print("DEVICE:", DEVICE)
print("FILEPATH:", FILEPATH)
print("OPERATION:", OPERATION)
print("NUM_HIDDEN:", NUM_HIDDEN)

########################################################################################################################
# Load data
print('# Load data')
########################################################################################################################

dataset_file = FILEPATH + 'dataset_RSICD.h5'
# dataset_file = r"/media/george/Data/RS data/dataset_RSICD.h5"
print('dataset_file:', dataset_file)
captions = get_data_from_h5_bert(dataset_file)
print("'captions' shape", captions.shape)

########################################################################################################################
# Captions preparation
print('# Captions preparation')
########################################################################################################################

new_captions = captions.reshape(-1)
print("'captions' shape", new_captions.shape)
print(new_captions[1337])

new_captions = [remove_punctuation(str(c, 'utf-8')) for c in new_captions]

print("'captions' length", len(new_captions))
print(new_captions[1337])

########################################################################################################################
# Tokenization
print('# Tokenization')
########################################################################################################################

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
# Convert the string "granola bars" to tokenized vocabulary IDs
cap_ids = tokenizer.encode(new_captions[1337])
# Print the IDs
print('cap_ids', cap_ids)
# Convert the IDs to the actual vocabulary item
# Notice how the subword unit (suffix) starts with "##" to indicate
# that it is part of the previous string
print('cap_tokens', tokenizer.convert_ids_to_tokens(cap_ids))

cap_ids = torch.LongTensor(cap_ids)
# Print the IDs
print('cap_ids', cap_ids)
print('type of cap_ids', type(cap_ids))

########################################################################################################################
# Max token length
print('# Max token length')
########################################################################################################################

token_lens = []
for c in new_captions:
    toks = tokenizer.encode(c, max_length=128)
    token_lens.append(len(toks))

max_token_len = max(token_lens)
print('Max len:', max_token_len)

# import seaborn as sns
# sns.displot(token_lens)

########################################################################################################################
# Dataset and Dataloader
print('# Dataset and Dataloader')
########################################################################################################################

captions_dataset = Dataset(new_captions, tokenizer, max_len=max_token_len)
captions_dataloader = torch.utils.data.DataLoader(captions_dataset, batch_size=16, shuffle=False)
print(captions_dataset[1337])

########################################################################################################################
# Load pretrained BERT
print('# Load pretrained BERT')
########################################################################################################################

model = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# model.eval()

########################################################################################################################
# Get embeddings
print('Get embeddings')
########################################################################################################################

embeddings = get_embeddings(model, captions_dataloader, DEVICE, num_hidden_states=NUM_HIDDEN, operation=OPERATION)
print('Embeddings shape:', embeddings.shape)

########################################################################################################################
# Save embeddings
print('Save embeddings')
########################################################################################################################

out_file = FILEPATH + 'bert_' + DATASET + r'_embeddings_' + OPERATION + '_' + str(NUM_HIDDEN) + r'.h5'
save_embeddings_hdf5(out_file, embeddings.astype(np.float32))

with h5py.File(out_file, 'r') as ds:
    embeddings = ds['embeddings'][:]
    embeddings_norm = (embeddings - embeddings.mean()) / embeddings.std()
    print("max, min, mean, var, std:", np.max(embeddings), np.min(embeddings), np.mean(embeddings), np.var(embeddings),
          np.std(embeddings))
    print("Normalized max, min, mean, var, std:", np.max(embeddings_norm), np.min(embeddings_norm),
          np.mean(embeddings_norm), np.var(embeddings_norm), np.std(embeddings_norm))

print("\n\n\n")