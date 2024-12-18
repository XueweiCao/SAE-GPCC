import numpy as np
from sklearn.preprocessing import StandardScaler


def array2tensor(array, input_dim=600):
    m, n = array.shape
    length = m * n
    sequence = array.reshape(1, length)
    
    tensor = []
    ini_flag = 0
    end_flag = 0
    while True:
        end_flag = ini_flag + input_dim
        if ini_flag >= length:
            break
        else:
            if end_flag > length:
                batch_seq = sequence[0][ini_flag:length]
                padding_len = input_dim - (length - ini_flag)
                padding_seq = np.zeros(padding_len)
                batch_seq = np.hstack((batch_seq, padding_seq))
            else:
                padding_len = 0
                batch_seq = sequence[0][ini_flag:end_flag]
        tensor.append(batch_seq)
        ini_flag = end_flag

    return tensor, padding_len


def tensor2array(decoded, padding_len, input_dim=600):
    array = []
    for n in range(len(decoded)):
        decoded_seq = decoded[n]
        N = input_dim
        if n == (len(decoded) - 1):
            N -= padding_len
        for i in range(int(N)):
            bit = float(decoded_seq[i])
            bit = int(round(bit))
            array.append(bit)

    array = np.asarray(array)
    return array


def array_encoder(encoded, input_dim=600, output_dim=100):
    array = []
    for sequence in encoded:
        for i in range(output_dim):
            bit = float(sequence[i])
            array.append(bit)
    array = np.asarray([array])

    my_standard = StandardScaler()
    my_standard.fit(array)
    array = my_standard.transform(array)
    tensor, padding_len = array2tensor(array, input_dim=input_dim)

    return tensor, padding_len, my_standard



def array_decoder(decoded, padding_len, my_standard, input_dim=600, output_dim=100):
    array = []
    for sequence in decoded:
        for i in range(input_dim):
            bit = round(float(sequence[i]))
            array.append(bit)
    array = np.asarray(array)
    if padding_len != 0:
        array = array[:-padding_len]

    array = my_standard.inverse_transform([array])
    decoded = np.asarray(array).reshape(-1, output_dim)

    return decoded


def last_transform(encoded, output_dim=20):
    array = []
    for tensor in encoded:
        sequence = []
        for i in range(output_dim):
            bit = float(tensor[i])
            sequence.append(bit)
        array.append(sequence)

    array = np.array(array)
    return array
