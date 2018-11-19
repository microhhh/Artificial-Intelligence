from translator.Markov.statistic import *
from translator.Markov.normalize import *


if __name__ == '__main__':
    gen_init()
    generate_pinyin_to_hanzi()
    generate_start()
    generate_emission()
    generate_tramsition()