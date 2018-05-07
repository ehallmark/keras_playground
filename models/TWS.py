from IBWrapper import IBWrapper
from ib.ext.EClientSocket import EClientSocket

accountName = "DU226952"
callback = IBWrapper()
tws = EClientSocket(callback)
host = ""
port = 6969
clientId = 1

tws.eConnect(host,port,clientId) # connect to TWS

tws.eDisconnect() # disconnect