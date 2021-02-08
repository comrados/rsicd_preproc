import h5py
import json
import numpy as np
import torch
from torchvision import models
import time
import copy
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images / 255
        self.labels = labels

    def __getitem__(self, idx):
        return (idx,
                torch.from_numpy(self.images[idx].astype('float32')),
                torch.from_numpy(self.labels[idx].astype('float32')))

    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return torch.from_numpy(self.labels.astype('float32'))


def get_data_from_h5(file):
    with h5py.File(file, 'r') as ds:
        labels = ds['labels'][:]
        labels_one_hot = ds['labels_one_hot'][:]
        images = ds['images'][:]
        captions = ds['captions'][:]
        bow_repr = ds['bag_of_words'][:]
        bow_terms = ds['bow_terms'][:]
    return images, labels_one_hot, labels, captions, bow_repr, bow_terms


def get_splits(dataset):
    splits = {'train': [], 'test': [], 'val': []}
    for img in dataset['images']:
        splits[img['split']].append(img['imgid'])
    return splits


def get_embeddings(model, dataloader, device):
    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        batch_outputs = []

        for idx, x, y in dataloader:
            x = x.to(device)  # to gpu (if presented)
            y = y.to(device)  # to gpu (if presented)
            batch_outputs.append(model(x))

        output = torch.vstack(
            batch_outputs)  # (num_batches, batch_size, output_dim) -> (num_batches * batch_size, output_dim)

        return output.cpu()  # return to cpu (or do nothing)


def save_embeddings_hdf5(out_file, embeddings):
    with h5py.File(out_file, 'w') as hf:
        print("Saved as '.h5' file to", out_file)
        hf.create_dataset('embeddings', data=embeddings)


def train_model(model, device, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train': {'loss': [], 'acc': []}, 'val': {'loss': [], 'acc': []}}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idxs, inputs, labels in dataloaders[phase]:
                # print("Batch {}".format(idxs.numpy()))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, labs = torch.max(labels, 1)  # one-hot-vectors -> idx
                    _, preds = torch.max(outputs, 1)  # outputs -> argmax idx
                    loss = criterion(outputs, labs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labs.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def plot_history(hist):
    def subplot_history(ax, stat):
        ax.plot(hist['train'][stat], label='training')
        ax.plot(hist['val'][stat], label='validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid()
        ax.legend()

    plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(211)
    plt.title('Loss per epoch')
    subplot_history(ax1, 'loss')

    ax2 = plt.subplot(212)
    plt.title('Accuracy per epoch')
    subplot_history(ax2, 'acc')

    plt.tight_layout()
    plt.show()


########################################################################################################################
# Constants
print('# Constants')
########################################################################################################################

DATASET = 'RSICD'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FILEPATH = r'./data/'

print("DATASET:", DATASET)
print("DEVICE:", DEVICE)
print("FILEPATH:", FILEPATH)

########################################################################################################################
# Load data
print('# Load data')
########################################################################################################################

dataset_file = FILEPATH + 'dataset_RSICD.h5'
print('dataset_file:', dataset_file)
images, labels_one_hot, labels, captions, bow_repr, bow_terms = get_data_from_h5(dataset_file)
print("'images' shape", images.shape)
print("'labels_one_hot' shape", labels_one_hot.shape)
print("'captions' shape", captions.shape)

########################################################################################################################
# Get splits
print('# Get splits')
########################################################################################################################

json_file = FILEPATH + DATASET + r'.json'
print('json_file:', json_file)

with open(json_file, 'r') as fp:
    dataset = json.load(fp)

splits = get_splits(dataset)

val_idx = splits['val']
train_idx = splits['train']
test_idx = splits['test']

print("Length 'val_idx': " + str(len(val_idx)) + ", Length 'train_idx': " + str(
    len(train_idx)) + ", Length 'test_idx': " + str(len(test_idx)))

########################################################################################################################
# Splits' shapes
print("# Splits' shapes")
########################################################################################################################

if images.shape != (len(images), 3, 224, 224):
    images = np.rollaxis(images, 3, 1)  # (-1, 224, 224, 3) -> (-1, 3, 224, 224)

img_train = images[train_idx]
img_val = images[val_idx]
lab_train = labels_one_hot[train_idx]
lab_val = labels_one_hot[val_idx]

print("All:", images.shape, labels_one_hot.shape)
print("Train:", img_train.shape, lab_train.shape)
print("Validation:", img_val.shape, lab_val.shape)

########################################################################################################################
# Datasets and dataloaders
print('# Datasets and dataloaders')
########################################################################################################################

train_data = Dataset(img_train, lab_train)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

val_data = Dataset(img_val, lab_val)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

dataset_sizes = {'train': len(train_data), 'val': len(val_data)}
print(dataset_sizes)

all_data = Dataset(images, labels_one_hot)
all_dataloader = torch.utils.data.DataLoader(all_data, batch_size=100, shuffle=False)
print("First img idx, shape, label:", all_data[0][0], all_data[0][1].numpy().shape, all_data[0][2].numpy().shape)

########################################################################################################################
# Load pretrained ResNet18
print('# Load pretrained ResNet18')
########################################################################################################################

print(DEVICE)

resnet18 = models.resnet18(pretrained=True)
# resnet18.eval()

########################################################################################################################
# Modify default ResNet18 for fine-tuning
print('# Modify default ResNet18 for fine-tuning')
########################################################################################################################

class_num = len(labels)
print("Classes:", class_num)

# Freeze the gradients of all of the layers in the features (convolutional) layers
for param in resnet18.parameters():
    param.requires_grad = False

# replace last FC layer with new ones (requires_grad=True by default)
resnet18.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, class_num)
)  # requires_grad = True by default

resnet18.eval()

########################################################################################################################
# Fine-tuning
print('# Fine-tuning')
########################################################################################################################

resnet18_ft = resnet18.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(resnet18_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

resnet18_ft, hist = train_model(resnet18_ft, DEVICE, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                                dataset_sizes, num_epochs=25)

# plot_history(hist)

########################################################################################################################
# Save model
print('# Save model')
########################################################################################################################
model_path = FILEPATH + 'resnet18-fine-tuned_' + DATASET
print('model_path:', model_path)

torch.save(resnet18_ft, model_path)

########################################################################################################################
# Prepare for representation extraction
print('# Prepare for representation extraction')
########################################################################################################################

resnet18_ft = torch.load(model_path)

# that's unnecessary, just locks all parameters (sets untrainable)
for param in resnet18_ft.parameters():
    param.requires_grad = False

resnet18_ft.fc = resnet18_ft.fc._modules['0']

print(resnet18_ft)

########################################################################################################################
# Get embeddings
print('# Get embeddings')
########################################################################################################################

ft_embeddings = get_embeddings(resnet18_ft, all_dataloader, DEVICE).numpy()
print('Output shape', ft_embeddings.shape)
print("max, min, mean, var, std:", np.max(ft_embeddings), np.min(ft_embeddings), np.mean(ft_embeddings),
      np.var(ft_embeddings), np.std(ft_embeddings))

########################################################################################################################
# Save embeddings
print('# Save embeddings')
########################################################################################################################

out_file = FILEPATH + 'resnet18_' + DATASET + r'_embeddings_ft.h5'
print('out_file:', out_file)

save_embeddings_hdf5(out_file, ft_embeddings)

with h5py.File(out_file, 'r') as ds:
    embeddings = ds['embeddings'][:]
    embeddings_norm = (embeddings - embeddings.mean()) / embeddings.std()
    print("max, min, mean, var, std:", np.max(embeddings), np.min(embeddings), np.mean(embeddings), np.var(embeddings),
          np.std(embeddings))
    print("Normalized max, min, mean, var, std:", np.max(embeddings_norm), np.min(embeddings_norm),
          np.mean(embeddings_norm), np.var(embeddings_norm), np.std(embeddings_norm))

print("\n\n\n")
