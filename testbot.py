import operator
import random
import itertools

from player import Bot


def permutations(config):
    """Returns unique elements from a list of permutations."""
    return list(set(itertools.permutations(config)))


class GameBlackboard:
    def __init__(self):
        # self.arff_data = []
        self.games_played = 0
        self.player_stats = []


class TestBot(Bot):
    """A bot that does logical reasoning based on the known spies and the
    results from the mission sabotages."""

    game_blackboard = GameBlackboard()

    def onGameRevealed(self, players, spies):
        self.spies = spies
        self.player_mission_results = {p.name: 0 for p in players}
        self.player_spy_chances = {p.name: 0. for p in players}
        self.game_stats = GameStats()
        self.game_stats.players = players
        self.taboo = []
        self.configurations = permutations([True, True, False, False])  # 排列
        """this returns [(False, True, False, True), (True, True, False, False), 
        (False, True, True, False), (True, False, True, False), 
        (True, False, False, True), (False, False, True, True)]"""

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
            if self.use_tree():
                sorted_players = sorted(self.player_spy_chances, key=self.player_spy_chances.get)
            else:
                sorted_players = sorted(self.player_mission_results, key=self.player_mission_results.get)

            selected_players = [self]
            i = 0
            while len(selected_players) != count:
                selected_players = selected_players + [p for p in self.others() if p.name == sorted_players[i]]
                i += 1
            return selected_players

    def _select(self, configurations):
        """This is a hook for inserting more advanced reasoning on top of the
        maximal amount of logical reasoning you can perform."""
        return random.choice(configurations)

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
    def sabotage(self):
        return True



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
