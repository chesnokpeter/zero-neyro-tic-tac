from tictactoe import Map, Player
import json
from nn import NeuralNetwork

# map = Map()
# player1 = Player('игрок 1', 'X')
# player2 = Player('игрок 2', 'O')
# win = False
# current_pl = player1

# games = 100

# while games > 0:

#     while not win:
#         take = input('Введите число 1-9: ')
#         try:
#             take = int(take)
#         except:
#             continue
#         take -= 1
#         if not (-1 < take < 9):
#             continue

#         if map.get(take) != 0:
#             continue

#         map.place(take, current_pl.symbol)

#         if map.check_win(current_pl.symbol):
#             print(f'Выиграл {current_pl.name}!')
#             win = True
#             break

#         if map.is_full():
#             print('Ничья!')
#             break

#         current_pl = player2 if current_pl == player1 else player1

#     games -= 1
#     win = False



import random


def converter1(s):
    if s == 'X':
        return 1
    elif s == 'O': 
        return 2
    else: return 0


def converter2(s):
    if s == 'O':
        return 1
    elif s == 'X': 
        return 2
    else: return 0



def play_game(nn1: NeuralNetwork, nn2: NeuralNetwork):
    map = Map()
    current_nn = nn1
    symbols = {nn1: 'X', nn2: 'O'}
    converters = {nn1: converter1, nn2:converter2 }
    back = ''
    while True:
        board = map.dataset_map(converters[current_nn])
        move = current_nn.forward(board)
        move = move.index(max(move))
        print(move)
        print(map.printable_map())
        if back == move:
            return None
        if map.get(move) == ('X' or 'O'):
            continue
        map.place(move, symbols[current_nn])
        back = move

        if map.check_win(symbols[current_nn]):
            print(f'{symbols[current_nn]} wins!')
            return current_nn

        if map.is_full():
            return None

        current_nn = nn2 if current_nn == nn1 else nn1


def simulate_games(num_games):

    for i in range(num_games):
        nn1 = NeuralNetwork(input_size=9, hidden_size=random.randint(5, 20), second_hidden_size=random.randint(5, 20), output_size=9)
        nn2 = NeuralNetwork(input_size=9, hidden_size=random.randint(5, 20), second_hidden_size=random.randint(5, 20), output_size=9)

        with open("dataset.txt", "r") as dataser:
            training_data = [json.loads(line) for line in dataser]
        nn1.train(training_data, epochs=1000, learning_rate=random.randint(1, 10) / 100)
        nn2.train(training_data, epochs=1000, learning_rate=random.randint(1, 10) / 100)

        result = play_game(nn1, nn2)
        if not result:
            continue
        result.save(f'runs/{i}.pkl')


simulate_games(5)