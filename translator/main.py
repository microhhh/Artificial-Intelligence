# coding: utf-8

import sys
sys.path.append('..')

from translator.model.viterbi import *


in_filename = sys.argv[1]
out_filename = sys.argv[2]

if __name__ == '__main__':
    writer = open(out_filename, "w", encoding='UTF-8')
    for line in open(in_filename, encoding='UTF-8'):
        line = line.strip().split(' ')
        ob = []
        for i in line:
            i = correct(i).lower()
            ob.append(i)
        try:
            result = viterbi(observations=ob, path_num=2)
            print(''.join(result[0].path))
            writer.write(''.join(result[0].path) + '\n')
        except:
            print('KeyError')
            writer.write('KeyError')
            continue

        writer.flush()
    writer.close()
