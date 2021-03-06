# coding: utf-8
import os
import json

DATA = 'data'
DEFAULT = 'default'


class HMM():

    def __init__(self, ):
        current_dir = self.pwd()
        self.py2hz_dict = self.readjson(os.path.join(current_dir, 'hmm_pinyin_to_hanzi.json'))
        self.start_dict = self.readjson(os.path.join(current_dir, 'hmm_start.json'))
        self.emission_dict = self.readjson(os.path.join(current_dir, 'hmm_emission.json'))
        self.transition_dict = self.readjson(os.path.join(current_dir, 'hmm_transition.json'))

    def readjson(self, filename):
        with open(filename) as outfile:
            return json.load(outfile)

    def pwd(self):
        return os.path.dirname(os.path.abspath(__file__))

    def start(self, state):

        data = self.start_dict[DATA]
        default = self.start_dict[DEFAULT]

        if state in data:
            prob = data[state]
        else:
            prob = default
        return float(prob)

    def emission(self, state, observation):

        pinyin = observation
        hanzi = state

        data = self.emission_dict[DATA]
        default = self.emission_dict[DEFAULT]

        if hanzi not in data:
            return float(default)

        prob_dict = data[hanzi]

        if pinyin not in prob_dict:
            return float(default)
        else:
            return float(prob_dict[pinyin])

    def transition(self, from_state, to_state):

        data = self.transition_dict[DATA]
        default = self.transition_dict[DEFAULT]

        if from_state not in data:
            return float(default)

        prob_dict = data[from_state]

        if to_state in prob_dict:
            return float(prob_dict[to_state])

        if DEFAULT in prob_dict:
            return float(prob_dict[DEFAULT])

        return float(default)

    def get_states(self, observation):
        return [hanzi for hanzi in self.py2hz_dict[observation]]
