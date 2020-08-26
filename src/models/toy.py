class Para():
    def __init__(self):
        print('most general')

    def verb(self):
        print('general action')


class Rec(Para):
    def __init__(self, x):
        super().__init__()
        print('medium')

    def verb(self):
        print('med action')

class Sq(Rec):
    def __init__(self):
        super(Rec, self).__init__()
        print('most specific')

stuff = Sq()
stuff.verb()
