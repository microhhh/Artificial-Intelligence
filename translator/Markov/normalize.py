# coding: utf-8
import json

INIT_START_FILE = 'init_start.json'
INIT_EMISSION_FILE = 'init_emission.json'
INIT_TRANSITION_FILE = 'init_transition.json'
PINYIN_TABLE = '../data/pinyin_table.txt'

PY_TO_HZ_FILE = 'hmm_pinyin_to_hanzi.json'
START_FILE = 'hmm_start.json'
EMISSION_FILE = 'hmm_emission.json'
TRANSITION_FILE = 'hmm_transition.json'

PINYIN_NUM = 406.
HANZI_NUM = 6763.


def write_to_file(obj, filename):
    with open(filename, 'w', encoding='UTF-8') as outfile:
        data = json.dumps(obj, indent=4, sort_keys=True)
        outfile.write(data)


def read_from_file(filename):
    with open(filename, encoding='UTF-8') as outfile:
        return json.load(outfile)


def generate_pinyin_to_hanzi():
    data = {}
    for line in open(PINYIN_TABLE, encoding='UTF-8'):
        line = line.strip()
        ls = line.split('=')
        if len(ls) != 2:
            raise Exception('invalid format')
        py, chars = ls
        py = py.strip()
        chars = chars.strip()
        if len(py) > 0 and len(chars) > 0:
            data[py] = chars
            print(py)

    write_to_file(data, PY_TO_HZ_FILE)


def generate_start():
    data = {'default': 1, 'data': None}
    start = read_from_file(INIT_START_FILE)
    count = HANZI_NUM
    for hanzi in start:
        count += start[hanzi]

    for hanzi in start:
        start[hanzi] = start[hanzi] / count

    data['default'] = 1.0 / count
    data['data'] = start

    write_to_file(data, START_FILE)


def generate_emission():
    data = {'default': 1.e-200, 'data': None}
    emission = read_from_file(INIT_EMISSION_FILE)

    for hanzi in emission:
        num_sum = 0.
        for pinyin in emission[hanzi]:
            num_sum += emission[hanzi][pinyin]
        for pinyin in emission[hanzi]:
            emission[hanzi][pinyin] = emission[hanzi][pinyin] / num_sum

    data['data'] = emission
    write_to_file(data, EMISSION_FILE)


def generate_tramsition():
    data = {'default': 1.0 / HANZI_NUM, 'data': None}
    transition = read_from_file(INIT_TRANSITION_FILE)
    for c1 in transition:
        num_sum = HANZI_NUM  # 每个字都有初始的均等机会
        for c2 in transition[c1]:
            num_sum += transition[c1][c2]

        for c2 in transition[c1]:
            transition[c1][c2] = float(transition[c1][c2] + 1) / num_sum
        transition[c1]['default'] = 1.0 / num_sum

    data['data'] = transition
    write_to_file(data, TRANSITION_FILE)


if __name__ == '__main__':
    generate_pinyin_to_hanzi()
    generate_start()
    generate_emission()
    generate_tramsition()
