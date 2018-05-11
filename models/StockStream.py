import yaml
from models.TWSUtil import TWSController


# Step 1 - Initialize configuration and handlers

# 1.a Create config variables (should probably load everything from a config file)
conf = yaml.load(open('config.txt', 'r', encoding='utf8'))
account = conf['account']
host = ""
port = 6969
clientId = 1


# 1.b Define handlers and create handler map
def price_handler(data):
    print("Found data: ", data)


def size_handler(data):
    print("Found size: ", data)


handlers = {'tick_Price': price_handler, 'tick_Size': size_handler}

# Step 2 - TWSController

# 2.a Create TWSController instance
controller = TWSController(account, host, port, clientId, handlers)

# 2.b Connect
controller.connect()

# 2.c Whatever stock you want to stream
controller.stream_stock_data('AAPL', time_seconds=10)

# 2.d Disconnect
controller.disconnect()

# Step 3 - Check bank account
