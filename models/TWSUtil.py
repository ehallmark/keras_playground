



def create_stock(creator, symbol,
               secType='STK', exchange='SMART',
               currency='USD'):
    return creator.create_contract(symbol,secType,exchange,currency)