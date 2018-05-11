import pandas as pd
import time
from models.IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
import yaml
from models.TWSUtil import TWSController


# define handlers
def price_handler(data):
    print("Found data: ", data)


def size_handler(data):
    print("Found size: ", data)


conf = yaml.load(open('config.txt', 'r', encoding='utf8'))
account = conf['account']
host = ""
port = 6969
clientId = 1

print("Using account: ", account)

controller = TWSController(account, host, port, clientId, {'tick_Price': price_handler, 'tick_Size': size_handler})
controller.connect()
controller.stream_stock_data('AAPL', time_seconds=10)
controller.disconnect()

