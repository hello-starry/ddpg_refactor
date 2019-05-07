
class tmp():
    def __init__(self, var):
        self.var = var

    def test(self):
        a = self.var.copy()
        a /= 2
        print("a:",a)
        print("var:",self.var)

def main():
    t = tmp(6)
    t.test()

    a = 7
    b = a
    b /= 2
    print(a)

if __name__ == '__main__':
    main()