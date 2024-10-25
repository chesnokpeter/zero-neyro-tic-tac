from tictactoe import Map, Player
import json
map = Map()
player1 = Player('игрок 1', 'X')
player2 = Player('игрок 2', 'O')
win = False
current_pl = player1

def converter(s):
    if s == 'X':
        return 1
    elif s == 'O': 
        return 2
    else: return 0


while win == False:
    h_1 = []
    h_2 = []
    print('\n\n')
    print(map.printable_map())
    print(f'Сейчас ходит {current_pl.name} | {current_pl.symbol}')
    take = input('Введите число 1-9: ')
    try:
        take = int(take)
    except:
        continue
    take -= 1
    # if not (-1 < take < 9):
    #     continue

    if map.get(take) == ('X' or 'O'):
        continue

    h_1 = map.dataset_map(converter)

    map.place(take, current_pl.symbol)

    h_2 = map.dataset_map(converter)


    if map.check_win(current_pl.symbol):
        print(map.printable_map())
        print(f'Выиграл {current_pl.name}!')
        win = True
        break

    if map.is_full():
        print(map.printable_map())
        print('Ничья!')
        break

    with open('dataset.txt', 'a') as data:
        json.dump((h_1, h_2), data)
        data.write('\n') 

    current_pl = player2 if current_pl == player1 else player1


