# coding: utf-8

import pypinyin

def is_chinese(v):
    if len(v) == 0:
        return False
    return all('\u4e00' <= c <= '\u9fff' or c == '〇' for c in v)

def topinyin(s):
    #s都是汉字
    py_list = pypinyin.lazy_pinyin(s)
    result = []
    for py in py_list:

        if py == '〇':
            result.append('ling')
        else:
            result.append(py)

    return result