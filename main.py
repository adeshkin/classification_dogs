import yaml
import os
from tqdm import tqdm
import wandb
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from data.dataset import get_dls
from model import get_model


class Trainer:
    def __init__(self, params):
        self.params = params
        self.data_loaders = get_dls(params['data_dir'],
                                    params['splits'],
                                    params['batch_size'])
        self.model = get_model(params['model_name'],
                               params['num_classes'])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=params['lr'],
                                   momentum=params['momentum'])
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                step_size=params['step_size'],
                                                gamma=params['gamma'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        self.model.train()

        metrics = dict()
        metrics['loss'] = 0.0
        metrics['acc'] = 0
        for inputs, labels in tqdm(self.data_loaders['train']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, pred_labels = torch.max(outputs, 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics['loss'] += loss.cpu().detach()
            metrics['acc'] += torch.sum(pred_labels == labels.data) / labels.shape[0]

        for m in metrics:
            metrics[m] = metrics[m] / len(self.data_loaders['train'])

        return metrics

    def eval(self):
        self.model.eval()

        metrics = dict()
        metrics['loss'] = 0.0
        metrics['acc'] = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.data_loaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, pred_labels = torch.max(outputs, 1)

                metrics['loss'] += loss.cpu().detach()
                metrics['acc'] += torch.sum(pred_labels == labels.data) / labels.shape[0]

        for m in metrics:
            metrics[m] = metrics[m] / len(self.data_loaders['val'])

        return metrics

    def log_images(self):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.data_loaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, pred_labels = torch.max(outputs, 1)

                column_names = ['image', 'gt_label', 'pred_label']
                my_data = []
                images = inputs.cpu().numpy()
                for i in range(pred_labels.shape[0]):
                    my_data.append([wandb.Image(images[i]), labels[i], pred_labels[i]])
                val_table = wandb.Table(data=my_data, columns=column_names)
                wandb.log({'my_val_table': val_table})
                break

    def run(self):
        wandb.init(project=self.params['project_name'], config=self.params)
        os.makedirs(self.params['chkpt_dir'], exist_ok=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        self.model = self.model.to(self.device)
        for epoch in range(self.params['num_epochs']):
            train_metrics = self.train()
            self.lr_scheduler.step()
            val_metrics = self.eval()

            logs = {'train': train_metrics,
                    'val': val_metrics,
                    'lr': self.optimizer.param_groups[0]["lr"]}

            wandb.log(logs, step=epoch)
            self.log_images()

            current_acc = val_metrics['acc']
            if current_acc > best_acc:
                best_acc = current_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f"{self.params['chkpt_dir']}/{self.params['model_name']}.pth")


if __name__ == '__main__':
    with open(f'config/resnet18.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    trainer = Trainer(params)
    trainer.run()
