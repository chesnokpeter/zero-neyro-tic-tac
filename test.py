dataser = open("dataset.txt", "r")

while True:
    line = dataser.readline()
    if not line:
        break
    print(line)

# закрываем файл
dataser.close