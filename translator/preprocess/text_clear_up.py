# coding: utf-8
import os
from translator.utils import *

ARTICLE_DIR = '../data/sina_news_utf8'
SENTENCE_FILE = './sentence.txt'

def clear_sentences(content):
    content = content.replace(' ', '')
    content = content.replace('\t', '')
    sentences = []
    s = ''
    for char in content:
        if is_chinese(char):
            s += char
        else:
            if s is not '':
                sentences.append(s)
            s = ''
    sentences.append(s)
    return [s.strip() for s in sentences]


def extract_sentence():
    all_files = []
    for root, directories, filenames in os.walk(ARTICLE_DIR):
        for filename in filenames:
            p = os.path.join(ARTICLE_DIR, filename)
            all_files.append(p)

    sentence_out = open(SENTENCE_FILE, 'w', encoding='UTF-8')
    for file in all_files:
        print('process ' + file)
        with open(file, encoding='UTF-8') as out:
            content = out.read()
            sentences = clear_sentences(content)
            sentence_out.write('\n'.join(sentences) + '\n')
    sentence_out.close()


if __name__ == '__main__':
    extract_sentence()
