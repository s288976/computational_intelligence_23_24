import random
import math
from copy import deepcopy
import numpy as np
from game import Game, Move, Player
from tqdm import tqdm
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    
def make_hashable_board(board):
    # this method explores the board and coding it in that way:
    #   - n consecutive 0 are coding in negative number -n (e.g. 9 consecutive 0 are coding in -9)
    #   - n consecutive 1 are coding in positive number n (e.g. 7 consecutive 1 are coding in 7)
    #   - n consucetive -1 are conding in char located in position n in alphabet (e.g. 4 consecutive -1 are coding in d)
    hashable_board = str()
    zero_code = 0
    one_code = 0
    minus_one_code = 96
    item = board[0][0]
    for r in range (5):
        for c in range (5):
            if item != board[r][c]:
                if item == 0:
                    hashable_board += str(zero_code)
                    zero_code = 0
                elif item == 1:
                    hashable_board += str(one_code)
                    one_code = 0
                elif item == -1:
                    hashable_board += chr(minus_one_code)
                    minus_one_code = 96
            item = board[r][c]
            if r==4 and c==4:
                if item == 0:
                    zero_code -= 1
                    hashable_board += str(zero_code)
                elif item == 1:
                    one_code += 1
                    hashable_board += str(one_code)
                elif item == -1:
                    minus_one_code += 1
                    hashable_board += chr(minus_one_code)
            if item == 0:
                zero_code -= 1
            elif item == 1:
                one_code += 1
            elif item == -1:
                minus_one_code += 1
    return hashable_board

def switch_player (game):
    game.current_player_idx += 1
    game.current_player_idx %= 2
    
POPULATION_SIZE = 50
OFFSPRING_SIZE = 10
N_GAMES = 100
N_EVOLUTION = 1000

class GAPlayer(Player):
    # Genetic Algorithm Player
    def __init__(self, best_player_file="") -> None:
        super().__init__()
        if (best_player_file==""):
            self.matches = 0
            self.wins = 0
            self.elo = 1000             # used ELO rating to evaluate fitness
            self.reach_2400 = False
            self.strategy = defaultdict(tuple[tuple[int, int], int])
        else:
            try:
                f = open(best_player_file, 'rb')
                best_player = pickle.load(f)
                self.matches = best_player.matches
                self.wins = best_player.wins
                self.elo = best_player.elo             # used ELO rating to evaluate fitness
                self.reach_2400 = best_player.reach_2400
                self.strategy = best_player.strategy
                f.close()
            except:
                print("Error loading best player")
                self.matches = 0
                self.wins = 0
                self.elo = 1000             # used ELO rating to evaluate fitness
                self.reach_2400 = False
                self.strategy = defaultdict(tuple[tuple[int, int], int])

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        curr_board = game.get_board()
        hashable_board = make_hashable_board(curr_board)
        curr_player = game.get_current_player()
        hashable_board = hashable_board + "," + str(curr_player)        # dictionary key = coded board + "," + current player
        if self.strategy.get(hashable_board, None) != None and random.random() > 0.1:     # if key in dictionary -> use stored strategy (leave a 10% possibility to explore a random move)
            from_pos = self.strategy[hashable_board][0]
            move = self.strategy[hashable_board][1]
        else:                                                   # else, do a random move
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    
    def optimization (self, population_file=""):
        if (population_file==""):
            population = [GAPlayer() for _ in range(POPULATION_SIZE)]
        else:
            try:
                f = open(population_file, 'rb')
                population = pickle.load(f)
                f.close()
            except:
                print("Error loading population")
                population = [GAPlayer() for _ in range(POPULATION_SIZE)]
        x = []
        y = []
        self.evaluate_population(population)                    # players population play N_GAMES matches each other to evaluate their strenght
        for n in tqdm(range(N_EVOLUTION)):
            self.evolve_population(population)                  # delete OFFSPRING_SIZE players and generate new players
            self.evaluate_population(population)                # reevaluate players
            population.sort(key=lambda e: e.elo, reverse=True)
            x.append(n)
            y.append(population[0].elo)
        population.sort(key=lambda e: e.elo, reverse=True)
        print (population[0].elo)
        print (population[1].elo)
        print (population[2].elo)
        best_individual = population[0]
        # select best individual and set parameters
        self.matches = best_individual.matches
        self.wins = best_individual.wins
        self.elo = best_individual.elo
        self.reach_2400 = best_individual.reach_2400
        self.strategy = best_individual.strategy
        f = open ('GAPopulation.pckl', 'wb')
        pickle.dump(population, f)
        f.close()
        f = open ('bestPlayer.pckl', 'wb')
        pickle.dump(best_individual, f)
        f.close()
        plt.plot(x, y, color='red')
        plt.show()
    
    def evaluate_population(self, population):
        for n in range(N_GAMES):
            p0 = random.choice(population)      # chose 2 random player from population
            p1 = random.choice(population)
            game = Game()
            winner, trajectory = game.play_trajectory(p0, p1)      # play a match
            p0.matches += 1
            p1.matches += 1
            # calculate parameter to define new elo ratings after match
            diff_elo = abs(p0.elo - p1.elo)    
            if diff_elo > 400:
                diff_elo = 400
            expected = 1/((10**(diff_elo/400))+1)
            if winner == 0:
                # add every move made by the winner player (p0) into dictionary
                for state in trajectory:
                    hashable_board = make_hashable_board(state[0])+",0"         # dictionary key = coded board + "," + current player
                    if state[3] == 0:
                        p0.strategy[hashable_board] = (state[1], state[2])
                p0.wins += 1
                # calculate parameter to define new elo ratings after match
                if (not p0.reach_2400 or p0.matches<30):
                    if (p0.matches < 30):
                        k = 40
                    else:
                        k = 20
                else:
                    k = 10
                elo_change = k*(1-expected)
                # calculate new elo ratings for each player
                p0.elo += elo_change
                p1.elo -= elo_change
                if p0.elo >= 2400:
                    p0.reach_2400 = True
            else:
                # add every move made by the winner player (p1) into dictionary
                for state in trajectory:
                    hashable_board = make_hashable_board(state[0])+",1"         # dictionary key = coded board + "," + current player
                    if state[3] == 1:
                        p1.strategy[hashable_board] = (state[1], state[2])
                p1.wins += 1
                # calculate parameter to define new elo ratings after match
                if (not p1.reach_2400 or p1.matches<30):
                    if (p1.matches < 30):
                        k = 40
                    else:
                        k = 20
                else:
                    k = 10
                elo_change = k*(1-expected)
                # calculate new elo ratings for each player
                p0.elo -= elo_change
                p1.elo += elo_change
                if p1.elo >= 2400:
                    p1.reach2400 = True
    
    def evolve_population(self, population):
        population.sort(key=lambda e: e.elo, reverse=True)
        population = population[:(POPULATION_SIZE-OFFSPRING_SIZE)]
        for n in range(OFFSPRING_SIZE):
            ind = random.choice(population)                 # chose one random player
            mutation_probability = 0.1
            if random.random() < mutation_probability:
                # mutate
                new_ind = self.mutate(ind)                  # create a new player mutating chosen player
            else:
                # crossover
                ind2 = random.choice(population)            # chose a second random player
                new_ind = self.uniform_crossover(ind, ind2) # create a new player with uniform crossover between the 2 players
            population.append(new_ind)
        assert len(population) == POPULATION_SIZE, f"Error in evolve population {len(population)}"

    def mutate(self, ind):
        if len(ind.strategy) == 0:
            return ind
        new_ind = deepcopy(ind)
        new_ind.wins = 0
        new_ind.winner_rate = 0
        new_ind.matches = 0
        for key, value in new_ind.strategy.items():
            if random.random() < 0.3:
                #change slide
                possible_moves = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
                col = value[0][0]
                row = value[0][1]
                # remove all moves not permitted
                if col == 0:
                    possible_moves.remove(Move.LEFT)
                elif col == 4:
                    possible_moves.remove(Move.RIGHT)
                if row == 0:
                    possible_moves.remove(Move.TOP)
                elif row == 4:
                    possible_moves.remove(Move.BOTTOM)
                # remove old move from possible moves
                possible_moves.remove(value[1])
                new_ind.strategy[key] = (new_ind.strategy[key][0], random.choice(possible_moves))
        return new_ind

    def uniform_crossover (self, ind1, ind2):
        if len(ind1.strategy) == 0 and len(ind2.strategy) == 0:     # if player 1 and player 2 strategies are empty, return player 1
            return ind1
        elif len(ind1.strategy) == 0:                               # if player 1 strategy is empty, return player 2
            return ind2
        elif len(ind2.strategy) == 0:                               # if player 2 strategy is empty, return player 1
            return ind1
        new_ind = GAPlayer()
        new_ind.elo = np.average([ind1.elo, ind2.elo])
        for key, value in ind1.strategy.items():
            nrandom = random.random()
            if nrandom < 0.5:                                       # 50% add strategy of player 1 in new player strategy
                new_ind.strategy[key] = value
        for key, value in ind2.strategy.items():
            nrandom = random.random()
            if nrandom < 0.5:                                       # 50% add strategy of player 2 in new player strategy
                new_ind.strategy[key] = value
        return new_ind
    
class MCPlayer(Player):
    # Monte Carlo Player
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        choiced_move = self.choice_move(game)
        from_pos = choiced_move[0][0]
        move = choiced_move[0][1]
        return from_pos, move
    
    def choice_move(self, game):
        move_evaluation = defaultdict(int)
        # play 100 random games and evaluate next move based on game results
        for n in range (100):
            game_copy = deepcopy(game)
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            current_player = game_copy.get_current_player()
            while not (game_copy.move(from_pos[0], from_pos[1], move, current_player)):     # make first move
                from_pos = (random.randint(0, 4), random.randint(0, 4))
                move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            switch_player(game_copy)
            key = (from_pos, move)                                          # save position and move as key
            winner = game_copy.play(RandomPlayer(), RandomPlayer())         # play game
            # evaluate move
            if winner == current_player:
                move_evaluation[key] += 1
            else:
                move_evaluation[key] += -1
        move_evaluation = sorted(move_evaluation.items(), key=lambda e: e[1], reverse=True)
        # return best next move
        return move_evaluation[0]

class RLPlayer(Player):
    # Reinforcement Learning Player
    def __init__(self, namefile="") -> None:
        super().__init__()
        # try loading a previous elaboration
        if (namefile==""):
            self.board_moves_values = self.train()
        elif(namefile==None):
            self.board_moves_values = None
        else:
            try:
                f = open(namefile, 'rb')
                self.board_moves_values = pickle.load(f)
                f.close()
            except:
                print("Error loading reinforcement learning player")
                self.board_moves_values = None

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_board = game.get_board()
        hashable_current_board = make_hashable_board(current_board)         # get coded board (dictionary key)
        dict_move = self.board_moves_values[hashable_current_board]         # get value from dict -> dictionary move containig evaluations for every move
        if len(dict_move)>0:
            current_player = game.get_current_player()
            index = 0
            # if player is player 0 -> take move with highest value
            if current_player == 0:
                best_move_evaluated = sorted(dict_move.items(), key=lambda e: e[1], reverse=True)
            # if player is player 1 -> take move with lowest value
            else:
                best_move_evaluated = sorted(dict_move.items(), key=lambda e: e[1])
            while True:
                # best_move_evaluated = (list of items -> key=((col, row), slide, current_player), eval move)
                col = best_move_evaluated[index][0][0][0]
                row = best_move_evaluated[index][0][0][1]
                # check if is a current player move and if the cube is free or it's owned by current player
                if best_move_evaluated[index][0][2] == current_player and (current_board[row][col] == current_player or current_board[row][col] == -1):
                    break
                else:
                    index += 1    
            from_pos = best_move_evaluated[index][0][0]
            move = best_move_evaluated[index][0][1]
        else:
            #not evaluated, do a random move
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    
    def train(self):
        print ("Preparing the player...please, wait")
        board_moves_values = defaultdict(dict)
        dict_moves = defaultdict(float)
        epsilon = 0.001
        n_iter = 100_000
        for n in tqdm(range(n_iter)):
            sigma = (n/n_iter) ** 2             # sigma is always less than 1 and parabolic
            r_number = random.random()
            if (r_number > sigma):
                #exploration -> random players
                p0 = RandomPlayer()
                p1 = RandomPlayer()
            else:
                #exploitation -> rl player vs random
                if r_number < 0.5:
                    p0 = RLPlayer(None)
                    p0.board_moves_values = board_moves_values
                    p1 = RandomPlayer()
                else:
                    p1 = RLPlayer(None)
                    p1.board_moves_values = board_moves_values
                    p0 = RandomPlayer()
            g = Game()
            winner, trajectory = g.play_trajectory(p0, p1)
            reward = 1
            if winner == 1:
                reward = -1
            # if winner is player 0, reward is positive; otherwise is negative
            for state in trajectory:
                hashable_board = make_hashable_board(state[0])
                move = (state[1], state[2], state[3])           # move = (from_pos, slide, current_player)
                dict_moves[move] = dict_moves[move] + epsilon * (reward - dict_moves[move])      # save results in dictionary moves
                board_moves_values[hashable_board] = dict_moves    # join board and move
        # save results
        f = open ('rlplayer.pckl', 'wb')
        pickle.dump(board_moves_values, f)
        f.close()
        print ("Done! Ready to play!")
        return board_moves_values
    
class Tree():
    def __init__(self, player, move=[None, None, None], depth=0, children=None) -> None:
        if player == 1:
            self.eval = -math.inf
        else:
            self.eval = math.inf
        self.move = move
        self.depth = depth
        if children == None:
            self.children = []
        else:
            self.children = children
        self.alpha = -math.inf
        self.beta = math.inf

    def get_evaluation(self) -> int:
        return self.eval
    
    def add_children(self, child):
        self.children.append(child)
    
    def print_tree(self):
        print("NODE", self.depth, ":", self.move, " - eval =", self.eval, " children: ", len(self.children))
        for c in self.children:
            c.print_tree()

class MinmaxPlayer(Player):
    def __init__(self, depth) -> None:
        super().__init__()
        self.depth = depth

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        computed_move = self.simulate_move(game)
        from_pos = (computed_move[0], computed_move[1])
        move = computed_move[2]
        return from_pos, move
    
    def simulate_move(self, game: 'Game'):
        tree = Tree(game.get_current_player())
        best_reward, best_node = self.minmax(game, tree, True, game.get_current_player())
        tree.eval = best_reward
        tree.move = best_node.move
        return tree.move
    
    def minmax(self, game, parent, maximizing_player, n_minmax_player, alpha=-math.inf, beta=math.inf):
        current_player = game.get_current_player()
        if parent.depth == self.depth or game.check_winner() > -1:          # if tree end or game end
            reward = self.evaluate(game.check_winner(), game.get_board(), n_minmax_player)
            return reward, None

        if maximizing_player:
            max_eval = -math.inf
            best_node = None
            # iterate all possible moves
            for col in range(5):
                for row in range(5):
                    if not ((row > 0 and row < 4) and (col > 0 and col < 4)):
                        possible_moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]
                        for move in possible_moves:
                            game_copy = deepcopy(game)
                            if game_copy.move(col, row, move, current_player):                  # make move if allowed
                                node = Tree(current_player, [col, row, move], parent.depth+1)   
                                parent.add_children(node)
                                switch_player(game_copy)            # after move, switch player before call recoursive function
                                eval, _ = self.minmax(game_copy, node, False, n_minmax_player, alpha, beta)
                                node.eval = eval
                                if (eval > max_eval):
                                    max_eval = eval
                                    best_node = node
                                alpha = max(alpha, eval)
                            if beta <= alpha:
                                break
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
            return max_eval, best_node
        else:
            min_eval = math.inf
            best_node = None
            # iterate all possible moves
            for col in range(5):
                for row in range(5):
                    if not ((row > 0 and row < 4) and (col > 0 and col < 4)):
                        possible_moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]
                        for move in possible_moves:
                            game_copy = deepcopy(game)
                            if game_copy.move(col, row, move, current_player):                  # make move if allowed
                                node = Tree(current_player, [col, row, move], parent.depth+1)
                                parent.add_children(node)
                                switch_player(game_copy)            # after move, switch player before call recoursive function
                                eval, _ = self.minmax(game_copy, node, True, n_minmax_player, alpha, beta)
                                node.eval = eval
                                if (eval < min_eval):
                                    min_eval = eval
                                    best_node = node
                                beta = min(beta, eval)
                            if beta <= alpha:
                                break
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
            return min_eval, best_node

    def evaluate (self, is_winner, board, n_minmax_player):
        # count zero and one values on rows, column and diagonals
        consecutive_one = sum(np.count_nonzero(board == 1, axis=0) + np.count_nonzero(board == 1, axis=1)) + np.count_nonzero(np.diagonal(board) == 1) + np.count_nonzero(np.diagonal(np.flip(board, axis=1)) == 1)
        consecutive_zero = sum(np.count_nonzero(board == 0, axis=0) + np.count_nonzero(board == 0, axis=1)) + np.count_nonzero(np.diagonal(board) == 0) + np.count_nonzero(np.diagonal(np.flip(board, axis=1)) == 0)    
        # compute reward in case of no winner player
        if n_minmax_player == 0 :
            reward = consecutive_zero + (consecutive_one*-1)
        else:
            reward = (consecutive_zero*-1) + consecutive_one
        # if minmax player wins/loses, add/remove 100 extra points plus 1 point for every pieces not taken (-1 value) to the reward
        if is_winner == n_minmax_player:
            reward = 100 + 1 * np.count_nonzero(board == -1) + reward
        elif is_winner > -1 :
            reward = (100 + 1 * np.count_nonzero(board == -1))*-1 + reward
        return reward
    
def play_games (player1, player2, n_games):
    p0_wins = 0
    p1_wins = 0
    for n_games in tqdm(range(n_games)):
        g = Game()
        winner = g.play(player1, player2)
        if winner:
            p1_wins += 1
        else:
            p0_wins += 1
    print (f"P0 wins: {p0_wins}, P1 wins: {p1_wins}\n")

if __name__ == '__main__':
    random_player = RandomPlayer()
    minmax_player_1 = MinmaxPlayer(1)
    minmax_player_3 = MinmaxPlayer(3)
    rl_player = RLPlayer("rlplayer.pckl")
    mc_player = MCPlayer()
    ga_player = GAPlayer("bestPlayer.pckl")
    print (f"Player 0: Random Player vs Player 1: MinMax Player (with depth=1) - 100 Matches")
    play_games(random_player, minmax_player_1, 100)
    print (f"Player 0: Random Player vs Player 1: Reinforcement Learning Player - 100 Matches")
    play_games(random_player, rl_player, 100)
    print (f"Player 0: Random Player vs Player 1: Monte Carlo Player - 100 Matches")
    play_games(random_player, mc_player, 100)
    print (f"Player 0: Random Player vs Player 1: Genetic Algoritm Player - 100 Matches")
    play_games(ga_player, random_player, 100)
    print (f"Player 0: Reinforcement Learning Player vs Player 1: MinMax Player (with depth=1) - 100 Matches")
    play_games(rl_player, minmax_player_1, 100)
    print (f"Player 0: Reinforcement Learning Player vs Player 1: MinMax Player (with depth=3) - 100 Matches")
    play_games(rl_player, minmax_player_3, 100)
    print (f"Player 0: Monte Carlo Player vs Player 1: MinMax Player (with depth=1) - 100 Matches")
    play_games(mc_player, minmax_player_1, 100)
    print (f"Player 0: Monte Carlo Player vs Player 1: MinMax Player (with depth=3) - 100 Matches")
    play_games(mc_player, minmax_player_3, 100)
    print (f"Player 0: Genetic Algorithm Player vs Player 1: MinMax Player (with depth=1) - 100 Matches")
    play_games(ga_player, minmax_player_1, 100)
    print (f"Player 0: Genetic Algorithm Player vs Player 1: MinMax Player (with depth=3) - 100 Matches")
    play_games(ga_player, minmax_player_3, 100)
    print (f"Player 0: Reinforcement Learning Player vs Player 1: Monte Carlo Player - 100 Matches")
    play_games(rl_player, mc_player, 100)
    print (f"Player 0: Reinforcement Learning Player vs Player 1: Genetic Algorithm Player - 100 Matches")
    play_games(rl_player, ga_player, 100)
    print (f"Player 0: Monte Carlo Player vs Player 1: Genetic Algorithm Player - 100 Matches")
    play_games(mc_player, ga_player, 100)