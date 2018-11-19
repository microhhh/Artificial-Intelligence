# coding: utf-8

import sys
import json
from translator.utils import *
sys.path.append('.')


SENTENCE_FILE = '../preprocess/sentence.txt'
WORD_FILE = '../data/word.txt'
HANZI2PINYIN_FILE = '../data/pinyin_table.txt'

BASE_START = 'init_start.json'
BASE_EMISSION = 'init_emission.json'
BASE_TRANSITION = 'init_transition.json'


def writejson2file(data, filename):
    with open(filename, 'w', encoding='UTF-8') as outfile:
        data = json.dumps(data, indent=4, sort_keys=True)
        outfile.write(data)


def process_hanzipinyin(emission):
    ## ./hanzipinyin.txt
    print('read from test.txt')
    for line in open(HANZI2PINYIN_FILE, encoding='UTF-8'):
        line = line.strip()
        if '=' not in line:
            continue
        pinyin, hanzis = line.split('=')
        hanzis = hanzis.split(' ')
        hanzis = [hz for hz in hanzis]
        print(pinyin)
        for hanzi in hanzis:
            emission.setdefault(hanzi, {})
            emission[hanzi].setdefault(pinyin, 0)
            emission[hanzi][pinyin] += 1


def read_from_sentence_txt(start, emission, transition):
    ## ./result/sentence.txt
    print('read from sentence.txt')
    for line in open(SENTENCE_FILE, encoding='UTF-8'):
        line =line.strip()
        if len(line) < 2:
            continue
        if not is_chinese(line):
            continue

        ## for start
        start.setdefault(line[0], 0)
        start[line[0]] += 1

        ## for emission
        pinyin_list = topinyin(line)
        char_list = [c for c in line]

        for hanzi, pinyin in zip(char_list, pinyin_list):
            emission.setdefault(hanzi, {})
            emission[hanzi].setdefault(pinyin, 0)
            emission[hanzi][pinyin] += 1

        ## for transition
        for f, t in zip(line[:-1], line[1:]):
            transition.setdefault(f, {})
            transition[f].setdefault(t, 0)
            transition[f][t] += 1


def read_from_word_txt(start, emission, transition):
    ## ! 基于word.txt的优化
    print('read from word.txt')
    _base = 1000.
    _min_value = 2.
    for line in open(WORD_FILE, encoding='UTF-8'):
        line = line.strip()
        if '=' not in line:
            continue
        if len(line) < 3:
            continue
        ls = line.split('=')
        if len(ls) != 2:
            continue
        word, num = ls
        word = word.strip()
        num = num.strip()
        if len(num) == 0:
            continue
        num = float(num)
        num = max(_min_value, num / _base)

        if not is_chinese(word):
            continue

        ## for start
        start.setdefault(word[0], 0)
        start[word[0]] += num

        ## for emission
        pinyin_list = topinyin(word)
        char_list = [c for c in word]
        for hanzi, pinyin in zip(char_list, pinyin_list):
            emission.setdefault(hanzi, {})
            emission[hanzi].setdefault(pinyin, 0)
            emission[hanzi][pinyin] += num

        ## for transition
        for f, t in zip(word[:-1], word[1:]):
            transition.setdefault(f, {})
            transition[f].setdefault(t, 0)
            transition[f][t] += num


def gen_init():

    start = {}
    emission = {}
    transition = {}

    process_hanzipinyin(emission)

    read_from_sentence_txt(start, emission, transition)
    read_from_word_txt(start, emission, transition)

    writejson2file(start, BASE_START)
    writejson2file(emission, BASE_EMISSION)
    writejson2file(transition, BASE_TRANSITION)


if __name__ == '__main__':
    gen_init()
