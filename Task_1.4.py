import random
from itertools import combinations

class ChessPlayer:
    def __init__(self,n,a,e,t,i,s):
        self.name = n
        self.age = a
        self.elo = e
        self.tenacity = t
        self.isBoring = i
        self.score = s

    def info(self):
        print(f"{self.name} \t\t\t {self.age} \t\t {self.elo} \t\t {self.tenacity} \t\t {self.isBoring} \t\t {self.score}")


def simulateMatch(c1, c2):
    if abs(c1.elo-c2.elo) > 100:
        if c1.elo > c2.elo:
            c1.score += 2
        else:
            c2.score += 2

    elif ((c1.isBoring == True) or (c2.isBoring == True)) and abs(c1.elo-c2.elo) < 100:
        c1.score += 1
        c2.score += 1

    elif 50 < abs(c1.elo-c2.elo) < 100:
        if c1.elo > c2.elo:
            if (c2.tenacity*random.randint(1,10)) > c1.elo:
                c2.score += 2
            else:
                c1.score += 2
        else:
            if (c1.tenacity*random.randint(1,10)) > c2.elo:
                c1.score += 2
            else:
                c2.score += 2

    elif abs(c1.elo-c2.elo) < 50:
        if c1.tenacity > c2.tenacity:
            c1.score += 2
        else:
            c2.score += 2

c1 = ChessPlayer("Courage the Cowardly Dog", 25, 1000.39, 1000, False, 0)
c2 = ChessPlayer("Princess Peach", 23, 945.65, 50, True, 0)
c3 = ChessPlayer("Walter White", 50, 1650.73, 750, False, 0)
c4 = ChessPlayer("Rory Gilmore", 16, 1700.87, 500, False, 0)
c5 = ChessPlayer("Anthony Fantano", 37, 1400.45, 400, True, 0)
c6 = ChessPlayer("Beth Harmon", 20, 2500.34, 150, False, 0)

c = [c1,c2,c3,c4,c5,c6]
d = list(combinations(c,2))

for i in d:
    simulateMatch(i[0],i[1])

func = lambda input: input.score

for i in sorted(c, key=func, reverse=True):
    i.info()