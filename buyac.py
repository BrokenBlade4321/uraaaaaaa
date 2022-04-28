import time

import MetaTrader5 as mt5
import numpy as np
import time
from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from threading import Thread

doc=0#"tradedata.json"
def LogIn():
    global is_bought,is_first_buy,lot1,lot2,symbol1,symbol2,k,lenght
    mt5.initialize()
    if mt5.login(login="1111711", password="7w63XbCV", server='Open-Demo'):#54893420 yvifmy8m MetaQuotes-Demo
        #1111711 7w63XbCV Open-Demo
        print(1)
    else:
        if doc:
            with open("tradedata.json") as data:
                is_bought=bool(data["is_bought"])
                is_first_buy=bool(data["is_first_buy"])
                lot1=int(data["lot1"])
                lot2 = int(data["lot2"])
                symbol1 = data["symbol1"]
                symbol2 = data["symbol2"]
                k=float(data["k"])
                length=int(data["length"])
class trade_bot:
    def __init__(self,symbol1,symbol2,lenght,lot1, lot2, k):
        self.stat=[0,0]
        self.lenght=lenght
        self.symbol1=symbol1
        self.symbol2=symbol2
        self.k=k
        self.sigma_k=1.7
        self.lot1=lot1
        self.lot2=lot2
        self.is_bought=None
        self.is_first_buy = True
        self.get_data()
    def  sum_natural_number(self,n,k, y=None):
        _sum = 0
        if y:
            for i in range(1, n + 1):
                _sum += (i ** k)*y[i-1]
            return float(_sum)
        for i in range(1,n+1):
            _sum += i**k
        return float(_sum)
    def get_data(self):
        pre_spreadd1 = list(mt5.copy_rates_from_pos(self.symbol1, mt5.TIMEFRAME_M1, 0, self.lenght)["close"])
        pre_spreadd2 = list(mt5.copy_rates_from_pos(self.symbol2, mt5.TIMEFRAME_M1, 0, self.lenght)["close"])
        self.spread = [pre_spreadd1[i] - self.k * pre_spreadd2[i] for i in range(self.lenght)]
        self.A = np.array([[self.sum_natural_number(self.lenght, 4 - i - j) for j in range(3)] for i in range(3)])
        B = np.array([self.sum_natural_number(self.lenght, 2 - j, y=self.spread) for j in range(3)])
        X = linalg.inv(self.A).dot(B)
        self.linear = X[0] * ((self.lenght) * (self.lenght)) + X[1] * (self.lenght) + X[2]
        self.linear_list = [X[0] * (i * i) + X[1] * i + X[2] for i in range(self.lenght)]
        new_spread = [self.spread[i] - self.linear_list[i] for i in range(self.lenght)]
        sqr_sum = sum([new_spread[i] * new_spread[i] for i in range(self.lenght)])
        self.sigma = (sqr_sum / self.lenght - (sum(new_spread) / self.lenght) ** 2) ** 0.5*self.sigma_k
        plt.plot(self.spread)
        plt.plot(self.linear_list)
        plt.plot([i-self.sigma for i in self.linear_list])
        plt.plot([i+self.sigma for i in self.linear_list])
        plt.grid(True)
        plt.show()
    def long_update(self):
        self.spread.pop(0)
        self.spread.append(self.get_last_value())
        B = np.array([self.sum_natural_number(self.lenght, 2 - j, y=self.spread) for j in range(3)])
        X = linalg.inv(self.A).dot(B)
        self.linear_list = [X[0] * (i * i) + X[1] * i + X[2] for i in range(self.lenght)]
        new_spread = [self.spread[i] - self.linear_list[i] for i in range(self.lenght)]
        sqr_sum = sum([new_spread[i] * new_spread[i] for i in range(self.lenght)])
        self.sigma = (sqr_sum / self.lenght - (sum(new_spread) / self.lenght) ** 2) ** 0.5*self.sigma_k

    def make_buy_deal(self, symbol, lot):
        price = round(mt5.symbol_info_tick(symbol).ask, 5)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price * 0.85,
            "tp": 1.15 * price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        m = mt5.order_send(request)
        print(m)

    def make_sell_deal(self, symbol, lot):
        price = round(mt5.symbol_info_tick(symbol).bid, 5)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": 1.15 * price,
            "tp": 0.85 * price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        m = mt5.order_send(request)
        print(m)

    def short_update(self):
        self.spread[-1]=self.get_last_value()

    def get_last_value(self):
        now = time.time()
        last_tick1 = mt5.copy_rates_from(self.symbol1, mt5.TIMEFRAME_M1, now, 1)
        last_tick2 = mt5.copy_rates_from(self.symbol2, mt5.TIMEFRAME_M1, now, 1)
        return last_tick1['close'][0]-self.k * last_tick2['close'][0]

    def first_buy(self):  # make deal
        if self.spread[-1] - self.linear > self.sigma:
            self.make_buy_deal(self.symbol2, self.lot2)
            self.make_sell_deal(self.symbol1, self.lot1)
            self.is_bought = False
            self.is_first_buy = False
            self.stat[0] += 1
        elif self.spread[-1] - self.linear < -self.sigma:
            self.make_buy_deal(self.symbol1, self.lot1)
            self.make_sell_deal(self.symbol2, self.lot2)
            self.is_bought = True
            self.is_first_buy = False
            self.stat[1] += 1

    def second_buy(self):  # make deal
        if self.is_bought:
            if self.spread[-1] - self.linear > self.sigma:
                self.is_bought = False
                self.stat[0] += 2
                self.make_buy_deal(2*self.symbol2, 2*self.lot2)
                self.make_sell_deal(2*self.symbol1, 2*self.lot1)
        else:
            if self.spread[-1] - self.linear < -self.sigma:
                self.is_bought = True
                self.stat[1] += 2
                self.make_buy_deal(2*self.symbol1, 2*self.lot1)
                self.make_sell_deal(2*self.symbol2, 2*self.lot2)

    def trade(self):
        if self.is_first_buy:
            self.first_buy()
        else:
            self.second_buy()


LogIn()
if doc:
    bot = trade_bot(symbol1, symbol2, lenght, lot1, lot2, k)
    bot.is_bought = is_bought
    bot.is_first_buy = is_first_buy
else:
    bot = trade_bot("BANEP","BANE",50000,10, 80, 0.8)

start=time.time()
i=0
plt.style.use('dark_background')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)



def animate(i,bot):
    data = bot.spread
    xs = []
    ys = []



    ax1.clear()
    ax1.plot(data)
    ax1.plot(bot.linear_list)
    ax1.plot([i+bot.sigma for i in bot.linear_list])
    ax1.plot([i-bot.sigma for i in bot.linear_list])
    print(7)
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.title('Обновляемые графики в matplotlib')



def animat():
    ani = animation.FuncAnimation(fig, lambda x: animate(x, bot), interval=1000)

def trading():
    global bot,i
    while True:
        i += 1
        time.sleep(1)
        bot.trade()
        if i<60:
            bot.short_update()
        else:
            bot.long_update()
            i=0
        print(i)
        print(i)
child1 = Thread(target=trading)    # задаем дочерний поток №1, который осуществляет парсинг
child1.start()
#child2 = Thread(target=animat)    # задаем дочерний поток №1, который осуществляет парсинг
ani = animation.FuncAnimation(fig, lambda x: animate(x, bot), interval=1000)
plt.show()
