"""
CE811 Assignment 1: The Resistance
An assignment bot called POLYTHENE

@Yuxi Zhou
Registration No: 2004457
"""

# Bring packages onto the path
import os
import sys
from player import Bot
import random
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from loggerbot import LoggerBot
from collections import defaultdict
import itertools
from sklearn import tree

model = keras.models.load_model('../log_classifier_02')

max_sc_total = 8


def permutations(config):
    """Returns unique elements from a list of permutations."""
    return list(set(itertools.permutations(config)))


class GameBlackboard:
    def __init__(self):
        self.arff_data = []
        self.games_played = 0
        self.player_stats = []


class Yz18966(Bot):

    game_blackboard = GameBlackboard()

    def mission_total_suspect_count(self, team):
        return sum([self.failed_missions_been_on[player] for player in team])

    def onGameRevealed(self, players, spies):
        self.spies = spies
        self.player_mission_results = {p.name: 0 for p in players}
        self.player_spy_probs = {p.name: 0. for p in players}
        self.game_stats = GameStats()
        self.game_stats.players = players
        self.team = None
        self.taboo = []

        self.missions_been_on = {}
        for p in players:
            self.missions_been_on[p] = 0
        self.failed_missions_been_on = {}
        for p in players:
            self.failed_missions_been_on[p] = 0

        self.num_mission_voted_up_with_total_suspect_count = {}
        self.num_mission_voted_down_with_total_suspect_count = {}
        for p in players:
            self.num_mission_voted_up_with_total_suspect_count[p] = [0] * max_sc_total
            self.num_mission_voted_down_with_total_suspect_count[p] = [0] * max_sc_total
        self.training_feature_vectors = {}
        for p in players:
            self.training_feature_vectors[p] = []

        self.configurations = permutations([True, True, False, False])  # 排列

    def calc_player_probabilities_of_being_spy(self):
        # This loop could be made much more efficient if we push all player's input patterns
        # through the neural network at once, instead of pushing them through one-by-one
        probabilities = {}

        for p in self.game.players:
            # This list comprising the input vector must build in **exactly** the same way as
            # we built data to train our neural network - otherwise the neural network
            # is not bieng used to approximate the same function it's been trained to model.
            # That's why this class inherits from the class LoggerBot
            # so we can ensure that logic is replicated exactly.

            # Note:
            #  Bot has not attribute 'missions_been_on' ?  @Yuxi 2020-11-12 01:19:35
            #   Fixed by declare those attribute from onGameRevealed and onMissionCompleted  @Yuxi 2020-11-12 11:11:26

            # print("==========================")
            # print(self.missions_been_on)
            # print("test2")
            input_vector = [self.game.turn, self.game.tries, p.index, p.name, self.missions_been_on[p],
                            self.failed_missions_been_on[p]] + self.num_mission_voted_up_with_total_suspect_count[p] + \
                           self.num_mission_voted_down_with_total_suspect_count[p]
            input_vector = input_vector[4:]  # remove the first 4 cosmetic details, as we did when training the neural
            # network
            input_vector = np.array(input_vector).reshape(1, -1)
            # change it to a rank-2 numpy array, ready for input
            # to the neural network.
            output = model(input_vector)  # run the neural network
            output_probabilities = tf.nn.softmax(output, axis=1)
            # The neural network didn't have a softmax on the final layer, so I'll add the softmax step here manually.
            probabilities[p] = output_probabilities[0, 1]
            # this [0,1] pulls off the first row (since there is only one row) and the second column (which
            # corresponds to probability of being a spy; the first column is the probability of being not-spy)
        return probabilities  # This returns a dictionary of {player: spyProbability}

    """
    Let's do some logic here
    """

    def getSpies(self, config):
        """on entry, config is a tuple of length 4 booleans, e.g. (False, True, False, True)"""
        assert len(config) == 4
        assert all([type(c) is bool for c in config])
        """ returns the subset of others who config says are spies"""
        return [player for player, spy in zip(self.others(), config) if spy]

    def getResistance(self, config):
        assert len(config) == 4
        assert all([type(c) is bool for c in config])
        """ returns the subset of others who config says are resistance"""
        return [player for player, spy in zip(self.others(), config) if not spy]

    def _validateSpies(self, config, team, sabotaged):
        """find which members of "team" are labelled as spies according
        to the config (boolean, boolean, boolean, boolean)"""
        spies = [s for s in team if s in self.getSpies(config)]
        """If there are more spies in our config than the number of sabotages made 
        then return True, because this config is compatible with the sabotages made.  
        Otherwise it is not compatible, so return False."""
        return len(spies) >= sabotaged

    def _validateNoSpies(self, config, team):
        spies = [s for s in team if s in self.getSpies(config)]
        """returns True if this config says there are zero spies present on this team."""
        return len(spies) == 0

    def select(self, players, count):
        if self.spy:
            return [self] + random.sample([p for p in self.others() if p not in self.spies], count - 1)
        else:

            # team = []
            # # If there was a previously selected successful team, pick it!
            # if self.team:  # and not self._discard(self.team):
            #     team = [p for p in self.team if p.index != self.index and p not in self.spies]
            # # If the previous team did not include me, reduce it by one.
            # if len(team) > count - 1:
            #     team = self._sample([], team, count - 1)
            # # If there are not enough people still, pick another randomly.
            # if len(team) == count - 1:
            #     return [self] + team
            # Try to put together another team that combines past winners and not spies.
            # others = [p for p in players if p != self and p not in (set(team) | self.spies)]
            # return self._sample([self] + team, others, count - 1 - len(team))




            if self.use_tree():
                sorted_players = sorted(self.player_spy_probs, key=self.player_spy_probs.get)
            else:
                sorted_players = sorted(self.player_mission_results, key=self.player_mission_results.get)

            selected_players = [self]
            i = 0
            while len(selected_players) != count:
                selected_players = selected_players + [p for p in self.others() if p.name == sorted_players[i]]
                i += 1
            return selected_players

    def _sample(self, selected, candidates, count):
        while True:
            selection = selected + random.sample(candidates, count)
            if self._discard(selection):
                continue
            return selection
        # The selected team has been discarded, meaning there's a problem with
        # the selected candidates.
        assert False, "Problem in team selection."

    def use_tree(self):
        for player_stats in self.game_blackboard.player_stats:
            if player_stats.name in [p.name for p in self.game_stats.players] and player_stats.clf is None:
                return False

        return True

    def _acceptable(self, team):
        """Determine if this team is an acceptable one to vote for..."""
        current = [c for c in self.configurations if self._validateNoSpies(c, team)]
        return bool(len(current) > 0)

    def _discard(self, team):
        # Has a subset of the proposed team failed a mission before?
        for t in self.taboo:
            if set(t).issubset(set(team)):
                return True
        return False

    def vote(self, team):
        # As a spy, vote for all missions that include one spy!
        if self.spy:
            return len([p for p in team if p in self.spies]) > 0
        # Always approve our own missions.
        if self.game.leader == self:
            return True
        # As resistance, always pass the fifth try.
        if self.game.tries == 5:
            return True
        # If there's a known spy on the team.
        if set(team).intersection(self.spies):
            return False
        # Taboo list of past suspicious teams.
        if self._discard(team):
            return False
        # If I'm not on the team and it's a team of 3!
        if len(team) == 3 and not self.index in [p.index for p in team]:
            return False
        # Otherwise, just approve the team and get more information.
        return True

    def _vote(self, team):
        """This is a hook for providing more complex voting once logical
        reasoning has been performed."""
        return True

    def sabotage(self):
        # the logic here is a bit boring and maybe could be improved.
        return True

    ''' The 3 methods onVoteComplete, onGameRevealed, onMissionComplete
    will inherit their functionality from ancestor.  We want them to do exactly 
    the same as they did when we captured the training data, so that the variables 
    for input to the NN are set correctly.  Hence we don't override these methods
    '''

    # This function used to output log data to the log file.
    # We don't need to log any data any more so let's override that function
    # and make it do nothing...
    def onGameComplete(self, win, spies):

        pass

    def onMissionComplete(self, sabotaged):
        if self.spy:
            return
        # Keep track of the team if it's successful
        if not sabotaged:
            self.team = self.game.team
            return

        # Divide the team into known spies and suspects
        suspects = [p for p in self.game.team if p not in self.spies and p != self]
        spies = [p for p in self.game.team if p in self.spies]

        if sabotaged >= len(suspects) + len(spies):
            # We have more thumbs down than suspects and spies!
            for spy in [s for s in suspects if s not in self.spies]:
                self.spies.add(spy)
        else:
            # Remember this specific failed teams so we can taboo search.
            self.taboo.append([p for p in self.game.team if p != self])


class GameStats:
    def __init__(self):
        self.players = []
        self.teams = []
        self.votes = []
        self.leaders = []
        self.spies = []
        self.results = []
        self.resistance_won = False
        self.player_stats = []

    def create_tree_data(self):
        pass


# this class is used both globally in the GameBlackboard and locally in the GameStats
class PlayerStats:
    def __init__(self, name, tree_data, arff_data):
        self.name = name
        self.tree_data = [tree_data]
        self.tree_results = []
        self.arff_data = [arff_data]
        self.clf = None
        # this is used only locally during a game
        self.spy_chance_total = 0

        # future tips
        # did they always vote to sabotage if they were on a mission
        # did they sabotage on the first round
        # did they behave differently on the last mission
