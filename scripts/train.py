import os
import sys
import yaml
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import get_dl
from model import get_model
from utils import MetricMonitor


class Trainer:
    def __init__(self, params):
        self.params = params
        self.data_loaders = get_dl(params['data_dir'],
                                   params['splits'],
                                   params['img_size'],
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

    def train(self, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()

        stream = tqdm(self.data_loaders['train'])
        for inputs, labels in stream:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred_labels = torch.max(outputs, 1)
            loss_value = loss.cpu().detach() / labels.shape[0]
            acc_value = torch.sum(pred_labels == labels.data).cpu().detach() / labels.shape[0]
            metric_monitor.update('loss', loss_value)
            metric_monitor.update('accuracy', acc_value)
            stream.set_description(
                "Epoch: {epoch}. Training. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        metrics = {metric: value / len(self.data_loaders['train'])
                   for (metric, value) in metric_monitor.get_metrics().items()}
        return metrics

    def eval(self, epoch):
        self.model.eval()
        metric_monitor = MetricMonitor()

        stream = tqdm(self.data_loaders['val'])
        with torch.no_grad():
            for inputs, labels in stream:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, pred_labels = torch.max(outputs, 1)
                loss_value = loss.cpu().detach() / labels.shape[0]
                acc_value = torch.sum(pred_labels == labels.data).cpu().detach() / labels.shape[0]
                metric_monitor.update('loss', loss_value)
                metric_monitor.update('accuracy', acc_value)
                stream.set_description(
                    "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
                )

        metrics = {metric: value / len(self.data_loaders['val'])
                   for (metric, value) in metric_monitor.get_metrics().items()}

        return metrics

    def run(self, config_filename):
        wandb.init(project=self.params['project_name'], config=self.params)
        os.makedirs(self.params['chkpt_dir'], exist_ok=True)

        best_acc = 0.0
        self.model = self.model.to(self.device)
        for epoch in range(1, self.params['num_epochs']+1):
            train_metrics = self.train(epoch)
            self.lr_scheduler.step()
            val_metrics = self.eval(epoch)

            logs = {'train': train_metrics,
                    'val': val_metrics,
                    'lr': self.optimizer.param_groups[0]["lr"]}

            wandb.log(logs, step=epoch)

            current_acc = val_metrics['acc']
            if current_acc > best_acc:
                best_acc = current_acc
                torch.save(self.model.state_dict(), f"{self.params['chkpt_dir']}/{config_filename}_best.pth")
            else:
                torch.save(self.model.state_dict(), f"{self.params['chkpt_dir']}/{config_filename}_last.pth")


if __name__ == '__main__':
    config_filename = sys.argv[1]
    with open(f'config/{config_filename}.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    trainer = Trainer(params)
    trainer.run(config_filename)
