import numpy as np


def coo_transform(points, bit_num):
    points = np.round(points * (10 ** bit_num))
    min_coo = np.min(points, axis=0)
    points -= min_coo

    return min_coo, points


def coo2morton(points, depth, default_depth=True):
    if default_depth:
        max_coo = np.max(points, axis=0)
        max_coo = np.floor(np.log2(max_coo)) + 1
        coding_depth = int(np.max(max_coo, axis=0))
    else:
        coding_depth = depth

    morton = []
    for point in points:
        x = int(point[0])
        y = int(point[1])
        z = int(point[2])
        code = 0
        for b in range(coding_depth):
            code |= ((x >> b) & 1) << (3 * b + 2)
            code |= ((y >> b) & 1) << (3 * b + 1)
            code |= ((z >> b) & 1) << (3 * b)
        morton.append(code)

    return morton, coding_depth


def diff_encoding(morton):
    morton.sort()
    diff_code = []
    for i in range(len(morton)):
        if i != 0:
            code = morton[i] - morton[i-1]
        else:
            code = morton[i]
        diff_code.append(code)

    return diff_code


def diff2bin(diff_code):
    diff_code = np.asarray(diff_code)
    max_diff = float(np.max(diff_code, axis=0))
    max_bit = int(np.floor(np.log2(max_diff)) + 1)

    bin_code = []
    for code in diff_code:
        b_code = bin(int(code))[2:]
        code_len = len(b_code)
        zero_len = max_bit - code_len
        zero_list = [0 for _ in range(zero_len)] 
        bin_code.append(zero_list)

        bin_list = [int(bit) for bit in b_code]
        bin_code.append(bin_list)

    bin_code = sum(bin_code, [])
    return max_bit, bin_code


def bin2block(bin_code, block_bit=6):
    ini_index = 0
    length = len(bin_code)
    block_code = []

    while True:
        end_index = ini_index + block_bit
        if ini_index >= length:
            break

        if end_index >= length:
            padding_len = end_index - length
            end_index = length
            code = bin_code[ini_index:end_index]
            for i in range(padding_len):
                code.append(0)
        else:
            code = bin_code[ini_index:end_index]

        bin_list = [str(item) for item in code]
        bin_str = ''.join(map(str, bin_list))
        bin_int = int(bin_str, 2)
        block_code.append(bin_int)

        ini_index = end_index

    return block_code, padding_len


def block2bin(block_code, padding_len, block_bit=6):
    block_code = np.asarray(block_code[0])
    bin_code = []
    for i in range(len(block_code)):
        code = int(block_code[i])
        bitstream =[]
        b_code = bin(code)[2:]
        code_len = len(b_code)
        zero_len = block_bit - code_len
        zero_list = [0 for _ in range(zero_len)] 
        bitstream.append(zero_list)

        bin_list = [int(bit) for bit in b_code]
        bitstream.append(bin_list)
        bitstream = sum(bitstream, [])
        if i == (len(block_code) - 1):
            bitstream = bitstream[:(block_bit-padding_len)]

        bin_code.append(bitstream)

    bin_code = sum(bin_code, [])
    return bin_code


def bin2diff(bin_code, max_bit):
    ini_index = 0
    length = len(bin_code)
    diff_code = []
    while True:
        end_index = ini_index + max_bit
        if ini_index >= length:
            break

        code = bin_code[ini_index:end_index]
        bin_list  = [str(element) for element in code]
        bin_str = ''.join(map(str, bin_list))
        bin_int = int(bin_str, 2)
        diff_code.append(bin_int)

        ini_index = end_index

    return diff_code


def diff_decoding(diff_code):
    morton = []
    for i in range(len(diff_code)):
        if i != 0:
            code = diff_code[i] + morton[i-1]
        else:
            code = diff_code[i]
        morton.append(code)
    
    return morton


def morton2coo(morton, depth):
    points = []
    for code in morton:
        x = 0
        y = 0
        z = 0
        for b in range(depth):
            x |= ((code >> (3 * b + 2)) & 1) << b
            y |= ((code >> (3 * b + 1)) & 1) << b
            z |= ((code >> (3 * b)) & 1) << b

        point = [x, y, z]
        points.append(point)

    return points


def coo_reconstruct(points, min_coo, bit_num):
    points += min_coo
    points /= (10 ** bit_num)

    return points


def pc_encoder(points, bit_num, depth=None, default_depth=True):
    min_coo, trans_points = coo_transform(points, bit_num)

    morton, coding_depth = coo2morton(trans_points, depth, default_depth=default_depth)

    diff_code = diff_encoding(morton)

    max_bit, bin_code = diff2bin(diff_code)

    block_code, padding_len = bin2block(bin_code)

    return block_code, min_coo, max_bit, padding_len, coding_depth


def pc_decoder(block_code, padding_len, max_bit, min_coo, bit_num, depth):
    bin_code = block2bin(block_code, padding_len)

    diff_code = bin2diff(bin_code, max_bit)

    morton = diff_decoding(diff_code)
    
    points = morton2coo(morton, depth)

    re_points = coo_reconstruct(points, min_coo, bit_num)

    return re_points
