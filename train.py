import copy
import json
import os

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score, \
    matthews_corrcoef, f1_score, recall_score, precision_score
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class train:
    def __init__(self, root: str):
        self._root = root
        self._train_dir = os.path.join(self._root, 'train')
        self._test_dir = os.path.join(self._root, 'test')
        self._val_dir = os.path.join(self._root, 'val')
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._pretrained_size = 224
        self._batch_size = 8
        self._num_epochs = 50
        self._loss_fn = nn.CrossEntropyLoss()
        self._metrics = ['accuracy', 'loss',
                         'precision',
                         'recall',
                         'f1-score',
                         'roc_auc_score',
                         'cohen_kappa_score',
                         'matthews_corrcoef',
                         'confusion_matrix']
        self._model = None
        self._model_name = None
        self._dataloaders = None
        self._image_datasets = None
        self._optimizer = None
        self.create_dataset()
    def create_dataset(self):
        pretrained_means = [0.485, 0.456, 0.406]
        pretrained_stds = [0.229, 0.224, 0.225]
        data_transforms = {

            'train': transforms.Compose([
                transforms.Resize((self._pretrained_size, self._pretrained_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=pretrained_means,
                                     std=pretrained_stds)
            ]),
            'test': transforms.Compose([
                transforms.Resize((self._pretrained_size, self._pretrained_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=pretrained_means,
                                     std=pretrained_stds)

            ])
            ,
            'val': transforms.Compose([
                transforms.Resize((self._pretrained_size, self._pretrained_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=pretrained_means,
                                     std=pretrained_stds)

            ])}
        print("Initializing Datasets and Dataloaders...\n")
        # Create training and validation datasets
        self._image_datasets = {x: datasets.ImageFolder(os.path.join(self._root, x), data_transforms[x]) for x in
                                ['train', 'test', 'val']}
        # Create training and validation dataloaders
        self._dataloaders = {
            x: torch.utils.data.DataLoader(self._image_datasets[x], batch_size=self._batch_size, shuffle=True,
                                           num_workers=4,
                                           pin_memory=True) for x in ['train', 'test', 'val']}

    def train_epoch(self):

        train_loss, train_correct = 0.0, 0
        self._model.train()

        __labels = torch.tensor([0]).to(self._device)
        __predictions = torch.tensor([0]).to(self._device)

        for images, labels in self._dataloaders["train"]:
            images, labels = images.to(self._device), labels.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(images)
            loss = self._loss_fn(output, labels)
            loss.backward()
            self._optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()
            __labels = torch.cat((__labels, labels), 0)
            __predictions = torch.cat((__predictions, predictions), 0)

        return train_loss, __labels[1::], __predictions[1::], scores

    def valid_epoch(self, dataloader):
        valid_loss, val_correct = 0.0, 0
        self._model.eval()
        __labels = torch.tensor([0]).to(self._device)
        __predictions = torch.tensor([0]).to(self._device)

        for images, labels in dataloader:
            images, labels = images.to(self._device), labels.to(self._device)
            output = self._model(images)
            loss = self._loss_fn(output, labels)
            valid_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            val_correct += (predictions == labels).sum().item()
            __labels = torch.cat((__labels, labels), 0)
            __predictions = torch.cat((__predictions, predictions), 0)

        return valid_loss, __labels[1::], __predictions[1::], scores

    def create_model(self, model_name):
        self._model = timm.create_model(model_name, pretrained=True)
        if model_name == "ghostnet_050":
            self._model.classifier.out_features = 4
        elif model_name == "seresnet18":
            self._model.fc.out_features = 4
        elif model_name == "rednet26t":
            self._model.head.fc.out_features = 4
        elif model_name == "vovnet39a":
            self._model.head.fc.out_features = 4
        elif model_name == "inception_v4":
            self._model.last_linear.out_features = 4
        elif model_name == "darknet53":
            self._model.head.fc.out_features = 4
        else:
            print("Model is not defined by user!")
        self._model.to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def set_history(self):
        history = {'train': {
            'accuracy': [],
            'loss': [],
            'precision': [],
            'recall': [],
            'f1-score': [],
            'cohen_kappa_score': [],
            'matthews_corrcoef': [],
        }, 'val': {
            'accuracy': [],
            'loss': [],
            'precision': [],
            'recall': [],
            'f1-score': [],
            'cohen_kappa_score': [],
            'matthews_corrcoef': [],
        }, 'model_name': self._model_name, 'num_epochs': self._num_epochs, 'batch_size': self._batch_size}
        return history

    def writeJson(self, history):
        log_file = self._model_name + ".json"
        with open(log_file, "w") as outfile:
            json.dump(history, outfile)

    def train_model(self,model_name):
        self.create_model(model_name)
        history = self.set_history()
        best_acc = 0.0
        best_model_wts = None
        for epoch in range(self._num_epochs):
            train_loss, train_labels, train_predictions, train_scores = self.train_epoch()
            test_loss, test_labels, test_predictions, test_scores = self.valid_epoch(self._dataloaders["val"])
            for phase in ['train', 'val']:
                for metric in self._metrics:
                    if metric == "accuracy":
                        history[phase][metric].append(accuracy_score(train_labels.cpu(), train_predictions.cpu()))
                    elif metric == "loss":
                        if phase == 'train':
                            history[phase][metric].append(train_loss / len(self._dataloaders[phase]))
                        else:
                            history[phase][metric].append(test_loss / len(self._dataloaders[phase]))
                    elif metric == "precision":
                        history[phase][metric].append(
                            precision_score(train_labels.cpu(), train_predictions.cpu(), average='weighted'))

                    elif metric == "recall":
                        history[phase][metric].append(
                            recall_score(train_labels.cpu(), train_predictions.cpu(), average='weighted'))

                    elif metric == "f1-score":
                        history[phase][metric].append(
                            f1_score(train_labels.cpu(), train_predictions.cpu(), average='weighted'))

                    elif metric == "cohen_kappa_score":
                        history[phase][metric].append(cohen_kappa_score(train_labels.cpu(), train_predictions.cpu()))

                    elif metric == "matthews_corrcoef":
                        history[phase][metric].append(matthews_corrcoef(train_labels.cpu(), train_predictions.cpu()))
            # deep copy the model
            if history["val"]["accuracy"][epoch] > best_acc:
                print('Best Val Acc. has been updated: {:4f}'.format(history["val"]["accuracy"][epoch]))
                best_acc = history["val"]["accuracy"][epoch]
                best_model_wts = copy.deepcopy(self._model.state_dict())
            print('Epoch {:.1f} has been completed with {:.3f} validation accuracy'.format(epoch,
                                                                                           history["val"]["accuracy"][
                                                                                               epoch]))
        print('Best Val Acc.: {:4f}'.format(best_acc))
        self._model.load_state_dict(best_model_wts)
        model_path = "./" + self._model_name + ".pth"
        torch.save(best_model_wts, model_path)
        self.writeJson(history)