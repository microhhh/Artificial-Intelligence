# coding: utf-8

'''
从文章中提取句子，放到sentence.txt中
'''


import os
import sys
import pypinyin
import importlib
# sys.path = ['../..'] + sys.path
from translator.utils import *

ARTICLE_DIR = '../data/sina_news_utf8'
SENTENCE_FILE = './sentence.txt'


def extract_chinese_sentences(content):
    # content = util.as_text(content)
    content = content.replace(' ', '')
    # content = content.replace('\n', '')
    # content = content.replace('\r', '')
    content = content.replace('\t', '')
    sentences = []
    s = ''
    for c in content:
        if is_chinese(c):
            s += c
        else:
            sentences.append(s)
            s = ''
    sentences.append(s)

    return [s.strip() for s in sentences if len(s.strip()) > 1]

def gen_sentence():
    all_files = []
    for root, directories, filenames in os.walk(ARTICLE_DIR):

        for filename in filenames:
            p = os.path.join(ARTICLE_DIR, filename)
            if p.endswith('.txt'):
                all_files.append(p)
    mid_out = open(SENTENCE_FILE, 'w', encoding='UTF-8')
    for fp in all_files:
        print('process '+ fp)
        with open(fp, encoding='UTF-8') as out:
            content = out.read()
            sentences = extract_chinese_sentences(content)
            mid_out.write('\n'.join(sentences) + '\n')

    mid_out.close()


if __name__ == '__main__':
    gen_sentence()