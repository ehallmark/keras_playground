import numpy as np
import pandas as pd
from models.IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
from threading import Lock
import time


class TWSController:
    # controller class to abstract away some of the complexities of IBWrapper
    def __init__(self, account, host, port, client_id, handlers):
        self.callback = IBWrapper(handlers, cacheMarketData=False)
        self.callback.initiate_variables()
        self.handlers = handlers
        self.tws = EClientSocket(self.callback)
        self.account = account
        self.host = host
        self.port = port
        self.clientId = client_id
        self.creator = contract()
        self.ticker_id = 1
        self.lock = Lock()

    def connect(self):
        self.tws.eConnect(self.host, self.port, self.clientId)  # connect to TWS
        self.tws.setServerLogLevel(5)  # Set error output to verbose

    def disconnect(self):
        self.tws.eDisconnect()  # disconnect

    # create a stock contract for the given symbol
    def create_stock(self, symbol,
                     sec_type='STK', exchange='SMART',
                     currency='USD'):
        return self.creator.create_contract(symbol, sec_type, exchange, currency)

    def create_option(self):
        pass

    def create_forex(self, symbol,
                     sec_type='CASH', exchange='IDEALPRO',
                     currency='USD'):
        return self.create_stock(symbol,sec_type,exchange,currency)

    def create_future(self):
        pass

    # thread safe
    def increment_ticker_id(self):
        with self.lock:
            self.ticker_id = self.ticker_id + 1
            ticker_id = self.ticker_id
        return ticker_id

    # opens a market request for the given ticker_id with various options
    # returns the ticker_id associated with the market request
    def open_market_stream(self, contract, host="", snapshot=False):
        ticker_id = self.increment_ticker_id()
        print('Opening market stream: ', ticker_id)
        self.tws.reqMktData(ticker_id, contract, host, snapshot)
        return ticker_id

    # closes a market request given by ticker_id
    def close_market_stream(self, ticker_id):
        print('Closing market stream: ', ticker_id)
        self.tws.cancelMktData(ticker_id)

    # streams market data for time_seconds seconds or forever if time_seconds is less than 0
    def stream_market_data(self, contract, time_seconds=-1, host="", snapshot=False):
        ticker_id = self.open_market_stream(contract, host, snapshot)
        i = 0
        while time_seconds < 0 or i < time_seconds:
            time.sleep(1)
            if 'listener' in self.handlers:
                self.handlers['listener'](i)
            i = i + 1
        self.close_market_stream(ticker_id)

    def stream_stock_data(self, symbol, sec_type='STK', exchange='SMART', currency='USD',
                          time_seconds=-1, snapshot=False):
        self.stream_market_data(self.create_stock(symbol, sec_type, exchange, currency),
                                time_seconds, self.host, snapshot)

    def stream_forex_data(self, symbol, sec_type='CASH', exchange='IDEALPRO', currency='USD',
                          time_seconds=-1, snapshot=False):
        self.stream_market_data(self.create_forex(symbol, sec_type, exchange, currency),
                                time_seconds, self.host, snapshot)

# Helper dictionary to map tick_type to human readable
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
