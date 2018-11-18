# coding: utf-8
from __future__ import (print_function, unicode_literals)

import sys
sys.path.append('..')

from translator.model.hmm import *
from translator.model.viterbi import *

hmmparams = HMM()

for line in open('in.txt', encoding='UTF-8'):
    line = line.strip().split(' ')
    ob=[]
    for i in line:
        ob.append(i.lower())
    # print(ob)
    # result = viterbi(hmm_params=hmmparams, observations=('qing', 'hua', 'da', 'xue'), path_num = 5, log = True)
    try:
        result = viterbi(hmm_params=hmmparams, observations=ob, path_num = 5, log = True)
        # for item in result:
        print(''.join(result[0].path))
    except:
        print('KeyError')
        continue

