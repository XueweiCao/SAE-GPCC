import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from model.dataloader import get_dataloader
from utils.pc_utils import load_pcd
from utils.encoder_utils import pc_encoder
import utils.model_utils as mu
import utils.data_utils as du
    

def get_sequences(train_path):
    input = []
    my_standard = StandardScaler()
    pcd_names = os.listdir(train_path)
    pcd_names = sorted(pcd_names)

    tot_num = len(pcd_names)
    cur_num = 1
    print('Datasets loading start')
    for pcd_name in pcd_names:
        print('Loading pcd: [{}/{}]'.format(cur_num, tot_num))
        pcd_path = train_path + pcd_name
        _, points = load_pcd(pcd_path)
    
        bit_num = 2
        block_code, _, _, _, _ = pc_encoder(points, bit_num)

        block_code = [block_code]
        my_standard.fit(block_code)
        block_code = my_standard.transform(block_code)
        cur_num = cur_num + 1
        input.append(block_code)

    print('Datasets loaded successfully\n')
    input = [item for sublist in input for sequence in sublist for item in sequence]
    input = np.asarray(input).reshape(1, -1)
    input_tensor, _ = du.array2tensor(input)

    return input_tensor


def stacked_training(input_tensor, models, device, batch_size, names, num_epochs):
    input = input_tensor
    for i in range(4):
        dataloader = get_dataloader(batch_size, input, input, device)
        encoder = models[i]
        epochs = num_epochs[i] 
        print('{} encoder training start'.format(names[i]))
        encoder = mu.train_model(encoder, dataloader, epochs)
        print('{} encoder training finished\n'.format(names[i]))
        models[i] = encoder

        output = mu.test_model(encoder, input, device, 'encode')
        if i == 2:
            input, _, _ = du.array_encoder(output, input_dim=100)
        else:
            input, _, _ = du.array_encoder(output)
        input = np.array(input)
        input = input[0:-1,:]
    return models


def save_training(models, save_path, names):
    time = datetime.now()
    time = time.strftime('%H_%M_%S')
    save_path = save_path + time + '/'
    os.makedirs(save_path, exist_ok=True)
    for i in range(4):
        model_name = names[i]
        model_path = save_path + model_name + '.pth'

        model = models[i]
        torch.save(model, model_path)

    print('Train models saved in {}\n'.format(save_path))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
        print('Device used in testing: {}\n'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('Device used in testing: cpu\n')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    train_path = './datasets/train/'
    input_tensor = get_sequences(train_path)
    input_tensor = np.array(input_tensor)

    models = mu.get_models(device)
    batch_size = 4096
    names = ['pre', 'mid', 'post', 'last']
    num_epochs = [1000, 2000, 2500, 3000]
    models = stacked_training(input_tensor, models, device, 
                              batch_size, names, num_epochs)

    save_path = './run/train/'
    save_training(models, save_path, names)
