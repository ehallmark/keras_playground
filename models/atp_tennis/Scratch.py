

import pandas as pd


df = pd.DataFrame({'A': [0,1,2,3,4,5]})
df.to_hdf('test.hdf', 'scratch', mode='w')

df = pd.read_hdf('test.hdf', 'scratch')

print(df)
