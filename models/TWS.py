from ib.ext.EClientSocket import EClientSocket
import pandas as pd
import numpy as np
import time
from datetime import datetime
from models.IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
from ib.ext.ScannerSubscription import ScannerSubscription
import yaml
import io

conf = yaml.load(open('config.txt','r',encoding='utf8'))
callback = IBWrapper()
tws = EClientSocket(callback)
account = conf['account']
host = ""
port = 6969
clientId = 1

print("Using account: ",account)

tws.eConnect(host,port,clientId) # connect to TWS

tws.setServerLogLevel(5)           # Set error output to verbose

create = contract()                # Instantiate contract class
callback.initiate_variables(callback)

tws.reqAccountUpdates(1,account)

time.sleep(0.5)

df = pd.DataFrame(callback.update_AccountValue,
            columns = ['key', 'value', 'currency', 'accountName'])[:3]

t = callback.update_AccountTime

tws.reqAccountSummary(2,"All","NetLiquidation")
time.sleep(0.5)

pos = pd.DataFrame(callback.account_Summary,
             columns = ['Request_ID','Account','Tag','Value','Currency'])

print("Positions: ",pos);
tws.reqPositions()

time.sleep(0.5)

dat = pd.DataFrame(callback.update_Position,
                   columns=['Account','Contract ID','Currency','Exchange','Expiry',
                            'Include Expired','Local Symbol','Multiplier','Right',
                            'Security Type','Strike','Symbol','Trading Class',
                            'Position','Average Cost'])
print(dat)

print(df)
print("Time: ", t)

tws.eDisconnect() # disconnect