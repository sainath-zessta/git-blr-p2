import os

os.system("cls")  # use this for windows. change to os.system("clear") for linux

COLORS = {
    "black": "\u001b[30;1m",
    "red": "\u001b[31;1m",
    "green": "\u001b[32m",
    "yellow": "\u001b[33;1m",
    "blue": "\u001b[34;1m",
    "magenta": "\u001b[35m",
    "cyan": "\u001b[36m",
    "white": "\u001b[37m",
    "yellow-background": "\u001b[43m",
    "black-background": "\u001b[40m",
    "cyan-background": "\u001b[46;1m",
}
# You can add more colors and backgrounds to the dictionary if you like.


def colorText(fileName):
    with open(fileName, "r") as f:
        ascii = "".join(f.readlines())
        for color in COLORS:
            ascii = ascii.replace("[[" + color + "]]", COLORS[color])
        print(ascii)
        return True
    print(os.getcwd())


def colorSingleLine(line):
    for color in COLORS:
        line = line.replace("[[" + color + "]]", COLORS[color])
    print(line)
    return True
