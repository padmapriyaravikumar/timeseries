from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/ML471_S1_Datafile_Concept.csv")
print(df.head())

size=int(len(df)*0.8)
train=df[:size]
test=df[size:]
model=SARIMAX(train['Consumption'],order=(1,1,1),seasonal_order=(1,2,1,12))
model_fit=model.fit()
res=model_fit.forecast(len(test))
plt.plot(res,label='predicted values')
plt.plot(train['Consumption'],label='train values')
plt.plot(test['Consumption'],label='test values')
plt.title("SARIMA FOR electrical consumption")
plt.legend()
plt.show()

model=SARIMAX(train['Consumption'],order=(1,1,1),seasonal_order=(1,2,1,12),exog=train[['Festivals/Special_events']])
model_fit=model.fit()
res=model_fit.forecast(len(test), exog=test[['Festivals/Special_events']])
plt.plot(res,label='predicted values')
plt.plot(train['Consumption'],label='train values')
plt.plot(test['Consumption'],label='test values')
plt.title("SARIMAX FOR electrical consumption")
plt.legend()
plt.show()
