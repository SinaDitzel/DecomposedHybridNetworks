from device import device
import torch
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image


class ClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=256, num_classes=200, hidden_nodes=512, linear_classifier=True, p=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AvgPool2d((7, 7), stride=2, padding=0)  # ,
        self.model = nn.Sequential()

        if not linear_classifier:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )

            self.model.add_module("ReLU", nn.ReLU())
            self.model.add_module("Dropout", nn.Dropout(p=p))

            self.model.add_module(
                "layer 2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )
        else:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, num_classes, bias=True)
            )
            #self.model.add_module("Dropout", nn.Dropout(p=p))

            #self.model.add_module(
            #    "layer2", nn.Linear(hidden_nodes, num_classes, bias=True)
            #)

    def forward(self, x):
        x = self.model(x)
        return x


def train_model(model, epoch, train_loader, optimizer, criterion, lrsheduler, log, print_every=10, reconstruction=True,vae=False):
    model.train(True)
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader)):
        lrsheduler.adjust_learning_rate(optimizer, epoch, i)
        # get the inputs
        inputs, labels = data
        if reconstruction or vae:
            labels = labels[0].to(device), labels[1].to(device)
        else:
            labels = labels.to(device)
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # self.net.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # loss = self.criterion(torch.sigmoid(outputs), one_hot(labels, self.dataset.num_classes))
        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            print("benchmark/run_model.py:train model: loss is nan!")
            print(" inputs nan ?", torch.any(torch.isnan(inputs)), "max, min:", torch.max(inputs), torch.min(inputs),
                  torch.min(torch.abs(inputs)))
            print(" outputs nan ?", torch.any(torch.isnan(outputs)), "max, min:", torch.max(outputs),
                  torch.min(outputs), torch.min(torch.abs(outputs)))
            sys.exit()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.detach().mean().item()

        '''if ((reconstruction or vae) and (i % 5 == 0)):
            #print(outputs)
            _,reconstructed_imgs,_,_ = outputs
            #grid_img = torchvision.utils.make_grid(reconstructed_imgs[0, :20], nrow=5)
            isExist = os.path.exists('grid_examples_vae_train')
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs('grid_examples_vae_train')
            save_image(reconstructed_imgs[:24],f'grid_examples_vae_train/{i}.png')

            #plt.figure()
            #plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
            #plt.savefig(f'grid_examples_vae_train/{i}.png')
            #plt.close()
        '''

        if i % print_every == print_every - 1:  # print every _ mini-batches
            idx = epoch + i / 1000  # float(str(epoch)+"."+str(i//print_every ))
            log.add_data("loss/train", running_loss / (print_every - 1), idx)
            log.add_data("learningrate", lrsheduler.lr, idx)
            running_loss = 0.0

    print("loss/train", running_loss / (i - 1))


def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=1)


def predictive_entropy(mc_preds):
    # calculate entropy in the case of multiple samples.
    return entropy(np.mean(mc_preds, axis=0))


def save_conf_img(img,number_of_channels=6, img_nb=2,folder_path='./confidence_examples',correct_classification=False,img_index=0):
    fig, axs = plt.subplots(1, number_of_channels, figsize=(10, 10))
    save_folder_name = os.path.join(f'{folder_path}_{number_of_channels}', str(correct_classification))
    # Create the path if it does not exist
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    for i in range(number_of_channels):
        axs[i].imshow(img[img_nb, i].detach().cpu().numpy(), cmap='gray')
        axs[i].axis('off')

    plt.savefig(os.path.join(save_folder_name,str(img_index)+'.png'))
    plt.close()

def accuracy_classification(model, dataloader, num_classes, reconstruction=True, entropies_csv='entropies.csv',
                            save_reconstruction=False):
    model.train(False)
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    if reconstruction and save_reconstruction:
        current_model_name = entropies_csv.split('.csv')[0]
        current_model_name = current_model_name .split('_entropies')[0]
        if not os.path.exists(f'{current_model_name}_grid_examples'):
            os.makedirs(f'{current_model_name}_grid_examples')
    with torch.no_grad():
        all_predictions = []
        all_entropies = []
        batch_idx =0
        for data in tqdm(dataloader):
            batch_idx +=1
            inputs, labels = data
            if reconstruction:
                labels = labels[0]
            inputs, labels = inputs.to(device), labels.to(device)
            if reconstruction:
                outputs,reconstructed_imgs,_,_ = model(inputs)
                outputs = torch.mean(outputs, dim=0)
                if save_reconstruction and batch_idx % 20==0:
                    grid_img = torchvision.utils.make_grid(reconstructed_imgs[:10], nrow=5)
                    plt.figure()
                    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
                    plt.savefig(f'{current_model_name}_grid_examples/{batch_idx}.png')
                    plt.close()
            else:
                outputs = model(inputs)
            m = torch.nn.Softmax(dim=1)
            sofmx = m(outputs)
            entropies = entropy(sofmx.detach().cpu().numpy())
            all_entropies.extend(entropies.tolist())
            if reconstruction:
                predicted = torch.argmax(outputs, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)

            correct_false_predictions = predicted == labels
            all_predictions.extend(correct_false_predictions.detach().cpu().numpy().tolist())
            if save_reconstruction:
                #we will also save the data for the false and correct classifications.
                if inputs.shape[1]==4 or inputs.shape[1]==6:
                    #make sure that we have confidence maps for RG norm or LBP
                    img_nb =2
                    save_conf_img(img=inputs,number_of_channels=inputs.shape[1],img_nb=img_nb,correct_classification= correct_false_predictions[img_nb].item(),img_index=batch_idx)
            total += labels.size(0)

            correct += (predicted == labels).sum().cpu().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    df = pd.DataFrame({'predictions': all_predictions, 'entropies': all_entropies})

    if os.path.exists(entropies_csv):
        os.remove(entropies_csv)

    df.to_csv(entropies_csv)

    return (correct / float(total)), confusion_matrix / confusion_matrix.sum(1, keepdim=True)


def accuracy_vae(model, classifier, test_loader, num_classes, entropies_csv='entropies.csv'):
    model.train(False)
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)

    all_entropies =[]
    all_predictions =[]
    for data in tqdm(test_loader):
        inputs, labels = data
        labels = labels[0]  # we only need the target_classes here and not the reconstruction lables.

        inputs, labels = inputs.to(device), labels.to(device)
        img_encodings, _, _, _ = model(inputs)
        #img_encodings = torch.mean(img_encodings[0], dim=0)  # Are we sampling here??
        img_encodings = img_encodings.view(img_encodings.size(0), -1)  # Flatten Representations for the classifier
        outputs = classifier(img_encodings)
        m = torch.nn.Softmax(dim=1)
        sofmx = m(outputs)
        entropies = entropy(sofmx.detach().cpu().numpy())
        all_entropies.extend(entropies.tolist())
        predicted = torch.argmax(outputs, 1)

        correct_false_predictions = predicted == labels
        all_predictions.extend(correct_false_predictions.detach().cpu().numpy().tolist())

        total += labels.size(0)

        correct += (predicted == labels).sum().cpu().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    df = pd.DataFrame({'predictions': all_predictions, 'entropies': all_entropies})
    if os.path.exists(entropies_csv):
        os.remove(entropies_csv)

    df.to_csv(entropies_csv)

    return (correct / float(total)), confusion_matrix / confusion_matrix.sum(1, keepdim=True)

def vae_val(model, dataloader, save_reconstruction,criterion):
    model.train(False)
    with torch.no_grad():
        batch_idx = 0
        running_loss =0
        for data in tqdm(dataloader):
            batch_idx += 1
            inputs, labels = data
            labels = labels[0].to(device), labels[1].to(device)
            inputs = inputs.to(device)
            outs = model(inputs)
            _, reconstructed_imgs, _, _ = outs
            loss = criterion(outs, labels)
            running_loss += loss.detach().mean().item()
            # if save_reconstruction and batch_idx % 5 == 0:
                #grid_img = torchvision.utils.make_grid(reconstructed_imgs[0, :20], nrow=5)
            #     save_image(reconstructed_imgs[:24], f'grid_examples_vae/{batch_idx}.png')
    return running_loss


def train_classifier(model, train_loader, num_classes):
    model.train(False)
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)

    # Initialize Classifier, Criterion and Optimizer
    #Adhoc way to figure out the correct shape of the classifier. Should be updated later
    random_input, _ = next(iter(train_loader))

    random_input = random_input.to(device)
    z_features, _,_,_ = model(random_input)
    input_features = z_features.view(z_features.size(0), -1).shape[1]

    classifier = ClassificationModel(in_channels=input_features, num_classes=num_classes,
                                     linear_classifier=True).to(device)

    criterion = torch.nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, #TODO check  these Hyperparameters.
                                 weight_decay=0.0001)
    for epoch in range(10):
        for data in tqdm(train_loader):
            inputs, labels = data
            labels = labels[0] # we only need the target_classes here and not the reconstruction labels.

            inputs, labels = inputs.to(device), labels.to(device)
            img_encodings, _, _, _ = model(inputs)
            #img_encodings = torch.mean(img_encodings[0], dim=0)  # Are we sampling here??
            img_encodings = img_encodings.view(img_encodings.size(0),
                                               -1)  # Flatten Representations for the classifier
            outputs = classifier(img_encodings)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().cpu().item()

        print('Epoch Accuracy',correct / float(total))

    return classifier
    """
        for data in tqdm(test_loader):
        inputs, labels = data
        labels = labels[0]  # we only need the target_classes here and not the reconstruction lables.

        inputs, labels = inputs.to(device), labels.to(device)
        img_encodings, _, _, _ = model(inputs)
        #img_encodings = torch.mean(img_encodings[0], dim=0)  # Are we sampling here??
        img_encodings = img_encodings.view(img_encodings.size(0), -1)  # Flatten Representations for the classifier
        outputs = classifier(img_encodings)
        m = torch.nn.Softmax(dim=1)
        sofmx = m(outputs)
        entropies = entropy(sofmx.detach().cpu().numpy())
        all_entropies.extend(entropies.tolist())
        predicted = torch.argmax(outputs, 1)

        correct_false_predictions = predicted == labels
        all_predictions.extend(correct_false_predictions.detach().cpu().numpy().tolist())

        total += labels.size(0)

        correct += (predicted == labels).sum().cpu().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    # print(confusion_matrix)

    df = pd.DataFrame({'predictions': all_predictions, 'entropies': all_entropies})
    if os.path.exists(entropies_csv):
        os.remove(entropies_csv)

    df.to_csv(entropies_csv)

    return (correct / float(total)), confusion_matrix / confusion_matrix.sum(1, keepdim=True)

    """
def accuracy_multiTarget(model, dataloader, num_classes, class_threshold=0.5):
    model.train(False)
    correct = 0
    correct_per_class = torch.tensor([0] * num_classes).to(device)
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            outputs = outputs > class_threshold
            comp = outputs.float() == labels.float()
            correct += (torch.prod(comp, dim=1)).sum().cpu().item()  # all clases have to be predicted correct
            correct_per_class += torch.sum(comp, dim=0)
            total += labels.size(0)
    return (correct / float(total)), None  # TODO CONFUSIOM MATRIX: (correct_per_class/float(total))
