from helper import data_loaders
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

#TRAINING
def train(train_data, model, optimizer, criterion, device, vocab, batch_size, clip, 
          teacher_forcing_ratio):
    
    model.to(device)
    model.train()
    train_loss = 0.

    for batch_idx, (input_tensor, lenght_tensor, target_tensor, max_target_len) in tqdm(
        enumerate(data_loaders(data=train_data, vocab=vocab, batch_size=batch_size)),
        desc='Training',
        total= len(train_data) // batch_size,
        leave=True,
        ncols=80
    ):
        
        # move data to GPU
        input_tensor=input_tensor.to(device)
        lenght_tensor=lenght_tensor.to('cpu') # always on cpu
        target_tensor = target_tensor.to(device)

        #clear gradient
        optimizer.zero_grad()
        output = model(input_tensor, target_tensor, max_target_len, lenght_tensor, teacher_forcing_ratio)
        # shape it for CrossEntropyLoss
        output = output.view(-1, output.shape[-1])
        target_tensor = target_tensor.flatten()
        #compute loss
        loss_value = criterion(output, target_tensor)
        #backward
        loss_value.backward()
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        #one step with optimizer
        optimizer.step()
        #update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )
    return train_loss

def valid_one_epoch(valid_data, model, criterion, device, vocab, batch_size):
    """one epoh validation """
    with torch.no_grad():
        model.eval()
        valid_loss = 0.

        for batch_idx , (input_tensor, lenght_tensor, target_tensor, max_target_len) in tqdm(
        enumerate(data_loaders(data=valid_data, vocab=vocab, batch_size=batch_size)),
        desc='Validation',
        total = len(valid_data) // batch_size,
        leave=True,
        ncols=80,
        ):
            # move data to GPU
            input_tensor=input_tensor.to(device)
            lenght_tensor=lenght_tensor.to('cpu') # always on cpu
            target_tensor = target_tensor.to(device)

            # forward pass
            output = model(input_tensor, target_tensor, max_target_len, lenght_tensor)
            # shape it for CrossEntropyLoss
            output = output.view(-1, output.shape[-1])
            target_tensor = target_tensor.flatten()
            #compute loss
            loss_value = criterion(output, target_tensor)
            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )
    return valid_loss

def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, 15])

def optimize(data, model, optimizer, criterion, n_epochs, save_path, device, vocab, batch_size, clip, teacher_forcing,interactive_tracking=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else: 
        liveloss = None

    valid_loss_min = None
    logs = {}

    for epoch in range(1, n_epochs+1):
        
        train_loss = train(
            train_data=data['train'],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            vocab=vocab,
            batch_size=batch_size,
            clip=clip,
            teacher_forcing_ratio=teacher_forcing
        )

        valid_loss = valid_one_epoch(
            valid_data=data['valid'],
            model=model,
            criterion=criterion,
            device=device,
            vocab=vocab,
            batch_size=batch_size
        )

        print(train_loss)
        print(valid_loss)
        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        #save every 10 epoch
        if epoch % 10 == 0:
            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)
        
        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()