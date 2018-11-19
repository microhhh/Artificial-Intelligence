# coding: utf-8

import sys
from translator.model.viterbi import *

sys.path.append('..')

if __name__ == '__main__':
    writer = open("out.txt", "w", encoding='UTF-8')
    for line in open('in.txt', encoding='UTF-8'):

        line = line.strip().split(' ')
        ob = []
        for i in line:
            ob.append(i.lower())
        try:
            result = viterbi(observations=ob, path_num=5, log=True)
            print(''.join(result[0].path))
            writer.write(str(''.join(result[0].path)))
        except:
            print('KeyError')
            continue
        writer.flush()
    writer.close()
