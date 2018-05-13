import statsmodels as sm
from statsmodels.tsa import ar_model

dta = []
arma_mod20 = ar_model.AR(dta, (2,0)).fit(disp=False)

