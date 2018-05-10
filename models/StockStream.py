import pandas as pd
import time
from models.IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
import yaml
from models.TWSUtil import create_stock

tick_type = {0 : "BID SIZE",
             1 : "BID PRICE",
             2 : "ASK PRICE",
             3 : "ASK SIZE",
             4 : "LAST PRICE",
             5 : "LAST SIZE",
             6 : "HIGH",
             7 : "LOW",
             8 : "VOLUME",
             9 : "CLOSE PRICE",
             10 : "BID OPTION COMPUTATION",
             11 : "ASK OPTION COMPUTATION",
             12 : "LAST OPTION COMPUTATION",
             13 : "MODEL OPTION COMPUTATION",
             14 : "OPEN_TICK",
             15 : "LOW 13 WEEK",
             16 : "HIGH 13 WEEK",
             17 : "LOW 26 WEEK",
             18 : "HIGH 26 WEEK",
             19 : "LOW 52 WEEK",
             20 : "HIGH 52 WEEK",
             21 : "AVG VOLUME",
             22 : "OPEN INTEREST",
             23 : "OPTION HISTORICAL VOL",
             24 : "OPTION IMPLIED VOL",
             27 : "OPTION CALL OPEN INTEREST",
             28 : "OPTION PUT OPEN INTEREST",
             29 : "OPTION CALL VOLUME"}


# define handlers
def price_handler(data):
    print("Found data: ", data)

    
def size_handler(data):
    print("Found size: ", data)


conf = yaml.load(open('config.txt', 'r', encoding='utf8'))
callback = IBWrapper({'tick_Price': price_handler, 'tick_Size': size_handler})
callback.initiate_variables()

tws = EClientSocket(callback)
account = conf['account']
host = ""
port = 6969
clientId = 1

print("Using account: ", account)

tws.eConnect(host, port, clientId)   # connect to TWS
tws.setServerLogLevel(5)             # Set error output to verbose

create = contract()

tickerId = 1
contract_info = create_stock(create, 'NVDA')

tws.reqMktData(tickerId, contract_info, "", False)

for i in range(100):
    time.sleep(5)

    tick_Price = pd.DataFrame(callback.tick_Price,
                             columns = ['tickerId', 'field', 'price', 'canAutoExecute'])
    tick_Price["Type"] = tick_Price["field"].map(tick_type)

    tick_Size = pd.DataFrame(getattr(callback,'tick_Size',[]),
                             columns=["tickerId", "field", "size"])

    tick_Size["Type"] = tick_Size["field"].map(tick_type)
    print(len(tick_Price))
    print(len(tick_Size))

tws.cancelMktData(tickerId)

tws.eDisconnect()    # disconnect

