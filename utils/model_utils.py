import torch
import torch.nn as nn
import timm.scheduler
from model.stacked_autoencoder import get_simple_model, get_stacked_model


def train_model(model, dataloader, num_epochs, lr=0.001):
    critera = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                 t_initial=num_epochs,
                                                 lr_min=1e-5,
                                                 warmup_t=75,
                                                 warmup_lr_init=1e-4)
    
    train_loss_all = []
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        train_loss = 0.0

        model.train()
        num = 0 
        for step, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            src = batch[0]
            tar = batch[1]
            B, N = tar.shape

            optimizer.zero_grad()
            encoded, decoded = model(src)
            out = decoded

            loss = critera(out, tar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num += 1
        train_loss_all.append(train_loss / num)
        if (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}] Train Loss: {:.8f}'.format(
                    epoch+1, num_epochs, train_loss_all[-1]))
            
    return model


def test_model(model, input_tensor, device, mode):
    model.eval()

    result = []
    for sequence in input_tensor:
        sequence = torch.tensor(sequence, dtype=torch.float32).to(device)
        if mode == 'encode':
            res_seq = model.encoder(sequence)
        elif mode == 'decode':
            res_seq = model.decoder(sequence)
        else:
            print('mode input error')
            return None
        result.append(res_seq)

    return result


def get_models(device):
    models = []

    pre_encoder = get_stacked_model(device)
    models.append(pre_encoder)
    mid_encoder = get_stacked_model(device)
    models.append(mid_encoder)
    post_encoder = get_stacked_model(device)
    models.append(post_encoder)

    last_encoder = get_simple_model(device)
    models.append(last_encoder)
    return models
