# coding: utf-8
import pypinyin
import heapq


def is_chinese(v):
    if len(v) == 0:
        return False
    return all('\u4e00' <= c <= '\u9fff' or c == '〇' for c in v)


def topinyin(s):
    py_list = pypinyin.lazy_pinyin(s)
    result = []
    for py in py_list:

        if py == '〇':
            result.append('ling')
        else:
            result.append(py)

    return result


class Item(object):

    def __init__(self, score, path):
        self.__score = score
        self.__path = path

    @property
    def score(self):
        return self.__score

    @property
    def path(self):
        return self.__path

    def __lt__(self, other):
        return self.__score < other.score

    def __le__(self, other):
        return self.__score <= other.score

    def __eq__(self, other):
        return self.__score == other.score

    def __ne__(self, other):
        return self.__score != other.score

    def __gt__(self, other):
        return self.__score > other.score

    def __ge__(self, other):
        return self.__score >= other.score

    def __str__(self):
        return '< score={0}, path={1} >'.format(self.__score, self.__path)

    def __repr__(self):
        return self.__str__()


class PrioritySet(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []

    def put(self, score, path):
        assert (isinstance(path, list) == True)
        heapq.heappush(self.data, [score, Item(score, path)])
        while len(self.data) > self.capacity:
            heapq.heappop(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.data:
            yield item[1]

    def __str__(self):
        s = '[ \n'
        for item in self.data:
            s = s + '\t' + str(item[1]) + '\n'
        s += ']'
        return s

    def __repr__(self):
        return self.__str__()
