import os
import sys
import yaml
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from utils import ID2NAME

from dataset import get_dl
from model import get_model
from utils import MetricMonitor, set_seed


class Trainer:
    def __init__(self, params):
        self.params = params
        self.data_loaders = get_dl(params['data_dir'],
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
        stream = tqdm(self.data_loaders['train1'])
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
            metric_monitor.update('loss', loss_value.item())
            metric_monitor.update('accuracy', acc_value.item())
            stream.set_description(
                "Epoch: {epoch}. Training. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        return metric_monitor.get_metrics()

    def eval(self, epoch):
        self.model.eval()
        metric_monitor = MetricMonitor()
        stream = tqdm(self.data_loaders['dev'])
        with torch.no_grad():
            for inputs, labels in stream:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, pred_labels = torch.max(outputs, 1)
                loss_value = loss.cpu() / labels.shape[0]
                acc_value = torch.sum(pred_labels == labels.data).cpu() / labels.shape[0]
                metric_monitor.update('loss', loss_value.item())
                metric_monitor.update('accuracy', acc_value.item())
                stream.set_description(
                    "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
                )

        return metric_monitor.get_metrics()

    def test(self):
        self.model.eval()
        metric_monitor = MetricMonitor()
        stream = tqdm(self.data_loaders['val'])
        gt = []
        pred = []
        with torch.no_grad():
            for inputs, labels in stream:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)

                _, pred_labels = torch.max(outputs, 1)
                gt.extend(labels.cpu().numpy().tolist())
                pred.extend(pred_labels.cpu().numpy().tolist())

                acc_value = torch.sum(pred_labels == labels.data).cpu() / labels.shape[0]
                metric_monitor.update('accuracy', acc_value.item())
                stream.set_description(
                    "Test. {metric_monitor}".format(metric_monitor=metric_monitor)
                )

        # Overall accuracy
        overall_accuracy = metric_monitor.get_metrics()['accuracy']
        print(f'Accuracy of the network on the val set: {round(overall_accuracy, 3)}')

        # Confusion matrix
        conf_mat = confusion_matrix(gt, pred)
        print('Confusion Matrix')
        print('-' * 16)
        print(conf_mat, '\n')

        # Per-class accuracy
        class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
        print('Per class accuracy')
        print('-' * 18)
        for idx, accuracy in enumerate(class_accuracy):
            class_name = ID2NAME[idx]
            print('Accuracy of class %8s : %0.2f %%' % (class_name, accuracy))

    def run(self, config_filename):
        wandb.init(project=self.params['project_name'], config=self.params)
        os.makedirs(self.params['chkpt_dir'], exist_ok=True)
        set_seed()
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

            current_acc = val_metrics['accuracy']
            if current_acc > best_acc:
                best_acc = current_acc
                print(f'\nSaved best model with val_accuracy = {round(current_acc, 3)}\n')
                torch.save(self.model.state_dict(), f"{self.params['chkpt_dir']}/{config_filename}_best.pth")
            else:
                torch.save(self.model.state_dict(), f"{self.params['chkpt_dir']}/{config_filename}_last.pth")


if __name__ == '__main__':
    config_filename = sys.argv[1]
    with open(f'config/{config_filename}.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    trainer = Trainer(params)
    trainer.run(config_filename)
