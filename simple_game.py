from tictactoe import Map, Player

map = Map()
player1 = Player('игрок 1', 'X')
player2 = Player('игрок 2', 'O')
win = False
current_pl = player1

while win == False:
    print(map.printable_map())
    print(f'Сейчас ходит {current_pl.name} | {current_pl.symbol}')
    take = input('Введите число 1-9: ')
    try:
        take = int(take)
    except:
        continue
    take -= 1
    if not (-1 < take < 9):
        continue

    if map.get(take) != 0:
        continue

    map.place(take, current_pl.symbol)

    if map.check_win(current_pl.symbol):
        print(map.printable_map())
        print(f'Выиграл {current_pl.name}!')
        win = True
        break

    if map.is_full():
        print(map.printable_map())
        print('Ничья!')
        break

    current_pl = player2 if current_pl == player1 else player1