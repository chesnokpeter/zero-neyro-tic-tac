class Map:
    def __init__(self):
        self.map = [[' ' for _ in range(3)] for _ in range(3)]

    def place(self, index: int, symbol: str):
        row = index // 3
        col = index % 3
        self.map[row][col] = symbol
    
    def get(self, index: int):
        row = index // 3
        col = index % 3
        return self.map[row][col]
    
    def printable_map(self):
        map = ''
        num = 0
        for i in self.map:
            row = '| '
            for j, el in enumerate(i):
                num += 1
                if el == ' ':
                    el = num
                row += f'{el} | '
            map += f'-------------\n{row}\n'
        map += '-------------'
        return map

    def check_win(self, symbol: str):
        for row in self.map:
            if all(el == symbol for el in row):
                return True
        for col in range(3):
            if all(self.map[row][col] == symbol for row in range(3)):
                return True
        
        if all(self.map[i][i] == symbol for i in range(3)):
            return True
        if all(self.map[i][2 - i] == symbol for i in range(3)):
            return True
        return False
    
    def is_full(self):
        return all(self.map[row][col] != ' ' for row in range(3) for col in range(3))

    def dataset_map(self, converter):
        o = []
        for i in self.map:
            for j in i:
                o.append(converter(j))
        return o

class Player:
    def __init__(self, name: str, symbol: str):
        self.symbol = symbol
        self.win = False
        self.name = name

