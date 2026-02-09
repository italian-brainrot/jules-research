import torch
import torch.nn as nn
import copy
from gswa_experiment.utils import compute_per_sample_grads_and_loss, compute_batch_gsnr

class GSWAManager:
    def __init__(self, model):
        self.averaged_model = copy.deepcopy(model)
        for p in self.averaged_model.parameters():
            p.detach_()
            p.zero_()
        self.sum_weights = 0.0
        self.num_averaged = 0

    def update(self, model, weight=1.0):
        for p_avg, p_src in zip(self.averaged_model.parameters(), model.parameters()):
            p_avg.data = (p_avg.data * self.sum_weights + p_src.data * weight) / (self.sum_weights + weight)
        self.sum_weights += weight
        self.num_averaged += 1

def train(model, train_loader, val_loader, optimizer, device, epochs, mode='Adam', swa_start_epoch=None):
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': [], 'gsnr': []}

    swa_manager = None
    if mode in ['SWA', 'GSWA']:
        swa_manager = GSWAManager(model)
        if swa_start_epoch is None:
            swa_start_epoch = int(epochs * 0.75)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_gsnrs = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # For GSWA we need GSNR
            if mode == 'GSWA' and epoch >= swa_start_epoch:
                grads_dict, loss_val = compute_per_sample_grads_and_loss(model, x, y)
                gsnr = compute_batch_gsnr(grads_dict)
                epoch_gsnrs.append(gsnr)
                # Apply gradients manually
                for name, p in model.named_parameters():
                    p.grad = grads_dict[name].mean(dim=0)
                loss_item = loss_val
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                loss_item = loss.item()

            optimizer.step()
            total_loss += loss_item

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_acc = correct / total
        history['val_acc'].append(val_acc)

        # SWA/GSWA Updates
        if swa_manager and epoch >= swa_start_epoch:
            if mode == 'SWA':
                swa_manager.update(model, weight=1.0)
            elif mode == 'GSWA':
                # Weight is the average GSNR of the epoch
                weight = sum(epoch_gsnrs) / len(epoch_gsnrs) if epoch_gsnrs else 1.0
                swa_manager.update(model, weight=weight)
                history['gsnr'].append(weight)

    if swa_manager and swa_manager.num_averaged > 0:
        return swa_manager.averaged_model, history
    return model, history

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
