import torch
import numpy as np
import os
import csv
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from utils.pc_utils import load_pcd, save_pcd
from utils.encoder_utils import pc_encoder, pc_decoder
from utils.model_utils import test_model
from utils.evaluate_utils import get_BPP, get_PSNR
import utils.data_utils as du


def get_trained_models(option_path, names, device):
    models = []
    for model_name in names:
        model_path = option_path + model_name + '.pth'
        model = torch.load(model_path, weights_only=False).to(device)
        models.append(model)
    
    return models


def stacked_encoder(input, models, device):
    pre_standard = StandardScaler()
    pre_standard.fit(input)
    input = pre_standard.transform(input)
    input, pre_plen = du.array2tensor(input)

    my_standard = []
    my_standard.append(pre_standard)
    padding_len = []
    padding_len.append(pre_plen)

    for model in models:
        encoded = test_model(model, input, device, 'encode')
        if model == models[2]:
            input, plen, standard = du.array_encoder(encoded, input_dim=100)
        elif model == models[-1]:
            break
        else:
            input, plen, standard = du.array_encoder(encoded)
        my_standard.append(standard)
        padding_len.append(plen)

    return encoded, my_standard, padding_len


def stacked_decoder(input, models, device, my_standard, padding_len):
    for i in range(len(models)):
        model = models[i]
        decoded = test_model(model, input, device, 'decode')

        if i < 3:
            standard = my_standard[i]
            plen = padding_len[i]
        else:
            break

        if i == 0:
            input = du.array_decoder(decoded, plen, standard, input_dim=100)
        else:
            input = du.array_decoder(decoded, plen, standard)

    standard = my_standard[-1]
    plen = padding_len[-1]
    block_code = du.tensor2array(decoded, plen)
    block_code = standard.inverse_transform([block_code])

    return block_code


def stacked_test(points, bit_num, models, device):
    block_code, min_coo, max_bit, block_plen, coding_depth = pc_encoder(points, bit_num)
    input = np.asarray(block_code).reshape(1, -1)

    encoded, my_standard, padding_len = stacked_encoder(input, models, device)

    input = du.last_transform(encoded)
    my_standard = my_standard[::-1]
    padding_len = padding_len[::-1]
    models = models[::-1]

    re_block_code = stacked_decoder(input, models, device, my_standard, padding_len)
    re_points = pc_decoder(re_block_code, block_plen, max_bit, min_coo, bit_num, coding_depth)

    return encoded, re_points


def single_frame_test(pcd_path, re_pcd_path, bit_num, models, device):
    pcd, points = load_pcd(pcd_path)
    npoint, _ = points.shape
    encoded, re_points = stacked_test(points, bit_num, models, device)

    BitPerPoint = get_BPP(encoded, npoint)
    re_pcd = save_pcd(re_points, re_pcd_path)
    c2c_PSNR, c2p_PSNR = get_PSNR(pcd, re_pcd)

    return BitPerPoint, c2c_PSNR, c2p_PSNR


def datasets_test(test_path, res_path, bit_num, models, device):
    time = datetime.now()
    time = time.strftime('%H_%M_%S')

    res_path = res_path + time + '/'
    os.makedirs(res_path, exist_ok=True)
    re_path = res_path + time + '/pcd/'
    os.makedirs(re_path, exist_ok=True)

    csv_path = create_csv(res_path)

    pcd_names = os.listdir(test_path)
    pcd_names = sorted(pcd_names)

    tot_num = len(pcd_names)
    cur_num = 1
    print('Datasets test start')
    for pcd_name in pcd_names:
        print('Testing pcd: [{}/{}]'.format(cur_num, tot_num))
        id = pcd_name[:-4]
        res = [id]
        res.append(bit_num)

        pcd_path = test_path + pcd_name
        re_pcd_path = re_path + pcd_name

        BitPerPoint, c2c_PSNR, c2p_PSNR = single_frame_test(pcd_path, re_pcd_path, bit_num, models, device)
        res.append(BitPerPoint)
        res.append(c2c_PSNR)
        res.append(c2p_PSNR)

        save_csv(res, csv_path)
        cur_num = cur_num + 1
    print('Datasets test finished\n')

    print('Test results saved in {}\n'.format(csv_path))


def create_csv(res_path):
    csv_path = res_path + 'result.csv'
    csv_header = ['id', 'Code_Bit', 'BPP', 'C2C_PSNR', 'C2P_PSNR']
    save_csv(csv_header, csv_path)

    return csv_path


def save_csv(csv_line, csv_path):
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:    
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(csv_line)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
        print('Device used in testing: {}\n'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device(f'cpu')
        print('Device used in testing: cpu\n')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    option_path = './model/option/'
    names = ['pre', 'mid', 'post', 'last']
    models = get_trained_models(option_path, names, device)

    test_path = './datasets/test/'
    res_path = './run/test/'
    bit_num = 0
    datasets_test(test_path, res_path, bit_num, models, device)
