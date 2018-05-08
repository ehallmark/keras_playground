from ib.ext.EClientSocket import EClientSocket
import pandas as pd
import numpy as np
import time
from datetime import datetime
from IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
from ib.ext.ScannerSubscription import ScannerSubscription

callback = IBWrapper()
tws = EClientSocket(callback)
account = "DU226952"
host = ""
port = 6969
clientId = 1

tws.eConnect(host,port,clientId) # connect to TWS

tws.setServerLogLevel(5)           # Set error output to verbose

create = contract()                # Instantiate contract class
callback.initiate_variables()

tws.reqAccountUpdates(1,account)

time.sleep(0.5)

df = pd.DataFrame(callback.update_AccountValue,
            columns = ['key', 'value', 'currency', 'accountName'])[:3]

t = callback.update_AccountTime

tws.reqAccountSummary(2,"All","NetLiquidation")
pd.DataFrame(callback.account_Summary,
             columns = ['Request_ID','Account','Tag','Value','Curency'])[:2]

tws.reqPositions()

time.sleep(0.5)

dat = pd.DataFrame(callback.update_Position,
                   columns=['Account','Contract ID','Currency','Exchange','Expiry',
                            'Include Expired','Local Symbol','Multiplier','Right',
                            'Security Type','Strike','Symbol','Trading Class',
                            'Position','Average Cost'])
print(dat[dat["Account"] == account])

print(df)
print("Time: ", t)

tws.eDisconnect() # disconnect