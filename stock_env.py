from enum import Enum

STOCK = Enum('Stock', ('Google', 'Tencent', 'Baidu'))


def get_data(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    # the first line is the header
    lines = lines[1:]

    x = []
    for line in lines:
        line = line.split(',')[1]
        x.append(float(line))  # [1] is the opening price

    return x


# class name using camel convention
class StockEnv:
    def __init__(self, stock = STOCK.Google):
        if stock == STOCK.Google:
            path = './data/GOOG.csv'
        elif stock == STOCK.Baidu:
            path = './data/BIDU.csv'
        elif stock == STOCK.Tencent:
            path = './data/TCEHY.csv'
        else:
            print('Invalid argument!')
            raise SystemError()
        self.data = get_data(path=path)

        self.cash = 10000
        self.n_stock = 0
        self.count = 0
        self.price_curr = self.data[self.count]
        self.asset = self.cash + self.n_stock * self.price_curr
        self.done = False

    # action in [-1, 1], where -1 represents sell all while 1 represents buy all
    # return (state, reward, done)
    def step(self, action):
        assert -1 <= action <= 1

        # update the cash and n_stock
        # 0 < action means buy in stocks
        if 0 < action:
            outlay = self.cash * action
            n_in = int(outlay / self.price_curr)
            self.cash -= n_in * self.price_curr
            self.n_stock += n_in
        else:
            n_out = int(- self.n_stock * action)
            self.n_stock -= n_out
            self.cash += n_out * self.price_curr

        # update the price
        self.count += 1
        self.price_curr = self.data[self.count]
        if self.count == len(self.data) - 1:
            self.done = True

        # calculate the reward
        asset = self.cash + self.n_stock * self.price_curr
        reward = asset - self.asset
        self.asset = asset

        state = (self.cash, self.n_stock, self.price_curr)
        return state, reward, self.done


if __name__ == '__main__':
    env = StockEnv()
    for x in range(10):
        print ('Enter your action between -1 and 1: ')
        my_action = input()
        print('New info: ((cash, n_stock, current price), reward, done)')
        print(env.step(my_action))
    print('You have earned $%f just now.' % (env.asset - 10000))
