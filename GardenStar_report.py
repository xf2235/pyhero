from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
chrome_driver = "./chromedriver.exe" 
driver = webdriver.Chrome(chrome_driver, options=chrome_options)

button_group = driver.find_element_by_class_name('button-group')
a_all = button_group.find_elements_by_tag_name('a')
for a in a_all:
    if not a.get_attribute('href'):
        year = a.text

table = driver.find_element_by_id('state_table')
trs = table.find_elements_by_tag_name('tr')
text = ''
for tr in trs:
    td_text =''
    tds = tr.find_elements_by_tag_name('td')
    for td in tds:
        td_text += td.text + '|'   
    td_text += year
    text += td_text.replace('\n', ' (') + '\n'

with open('./152.txt', 'a+', encoding='utf-8') as f:
    f.write(text)


chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
chrome_driver = "./chromedriver.exe"
driver = webdriver.Chrome(chrome_driver, options=chrome_options)

example_wrapper = driver.find_element('id', 'example_wrapper')
ths= example_wrapper.find_element('tag name', 'thead').find_elements('tag name', 'th')
keys = [th.text for th in ths]
pages = example_wrapper.find_element('id', 'example_paginate').find_element('tag name', 'span').find_elements('tag name', 'a')
web_data = []
for page in pages:
    page.click()
    time.sleep(3)
    trs = example_wrapper.find_element('tag name', 'tbody').find_elements('tag name', 'tr')
    for tr in trs:
        tds = tr.find_elements('tag name', 'td')
        tds_data = [td.text for td in tds]
        web_data.append(tds_data)

data = pd.DataFrame(web_data, columns=keys)
print(data)
data.to_csv('./data.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r'rental_prices.xls'
data = pd.read_excel(path)  
print(data)

data_temp = pd.DataFrame(columns=['year', '0 BR', '1 BR', '2 BR', '3 BR', '4 BR'], index=[year for year in range(2006, 2022)])

for year in list(data_temp.index):
    data_temp.loc[year]['year'] = year
    for br in list(data_temp.columns)[1:]:
        k = data.loc[(data['year'] == year) & (data['County'] == 'Decatur County')][br]
        data_temp.loc[year][br] = int(k)

print(data_temp)
print(data_temp.loc[data_temp['year'] == 2021])

data_pic = data_temp[['0 BR', '1 BR', '2 BR', '3 BR', '4 BR']]

plt.plot(data_temp['year'], data_pic['0 BR'], linewidth=2, markersize=12, label="0 BR")
plt.plot(data_temp['year'], data_pic['1 BR'], linewidth=2, markersize=12, label="1 BR")
plt.plot(data_temp['year'], data_pic['2 BR'], linewidth=2, markersize=12, label="2 BR")
plt.plot(data_temp['year'], data_pic['3 BR'], linewidth=2, markersize=12, label="3 BR")
plt.plot(data_temp['year'], data_pic['4 BR'], linewidth=2, markersize=12, label="4 BR")
plt.legend()   
plt.savefig('data.png', format='png', dpi=200)    
plt.show()            
plt.close()

data_temp.to_csv('data_dispose.csv', index=False)


path = r'data.csv'
data = pd.read_csv(path) 
data_temp = data[['Year', 'Violent', 'Murder', 'Rape', 'Robbery', 'Assault', 'Property', 'Burglary', 'Larceny', 'Auto']]

data_temp.sort_values(by="Year", inplace=True, ascending=True)
data_temp.fillna(0, inplace=True)  


data_need = data_temp[['Year', 'Burglary']]
df = pd.DataFrame([[2019, 231.5], [2020, 215.6], [2021, 204.0], [2006, 0]], columns=list(['Year', 'Burglary']))      # 构造一个新的dataframe用于存储填补的数据
dp = data_need.append(df, ignore_index=True)    

data_test = dp[dp["Year"] >= 2006]
data_test.sort_values(by="Year", inplace=True, ascending=True)
data_test.to_csv('data_temp.csv', index=False)

data_one = pd.read_csv("data_temp.csv")
data_two = pd.read_csv("data_dispose.csv")
data_two['Burglary'] = data_one['Burglary']


data_thr = pd.read_csv("interest.csv")

data_two['interest'] = data_thr['interest']
print(data_two)
data_two.to_csv('data_dispose3.csv', index=False)

import pandas as pd
import  numpy as np
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split

data = pd.read_csv('data_dispose3.csv')

data_x = data[['year', 'Burglary', 'interest']]
y = data['0 BR']   # 获取租金数据

X_train, X_test, Y_train, Y_test = train_test_split(data_x, y, train_size=.70)  

model = LR()   
model.fit(X_train, Y_train)   
score = model.score(X_test, Y_test) 
print(score)

y_pred = model.predict(X_test) 
print(y_pred)
print(model.predict([[2022, 160, 0.11]]), model.predict([[2023, 140, 0.09]]))   

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

data = pd.read_csv('data_dispose2.csv')
print(data)

data_temp = data[['year', '0 BR', 'Burglary']]
data0 = data['0 BR'].values.astype(float)
data1 = data['1 BR'].values.astype(float)
data2 = data['2 BR'].values.astype(float)
data3 = data['3 BR'].values.astype(float)
data4 = data['4 BR'].values.astype(float)
print(data0, data1, data2, data3, data4)

test_data_size = 4  
train_data = data0[:-test_data_size]
test_data = data0[-test_data_size:]

scaler = MinMaxScaler(feature_range=(-1, 1)) 
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))  
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1) 

train_window = 3  
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)  
print(train_inout_seq)

model = LSTM()
loss_function = nn.MSELoss()   
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
print(model)


epochs = 150 
for i in range(epochs):
    for seq, labels in train_inout_seq:  
        optimizer.zero_grad()  
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                             torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)  
        single_loss = loss_function(y_pred, labels)  
        single_loss.backward() 
        optimizer.step()

    if i % 10 == 1:  
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 3
test_inputs = train_data_normalized[-train_window:].tolist()  
print(test_inputs)

model.eval()  
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        test_inputs.append(model(seq).item())

print(test_inputs[fut_pred:])

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print(actual_predictions)
d1 = actual_predictions[0][0]
d2 = actual_predictions[1][0]
d3 = actual_predictions[2][0]
d = [int(d1), int(d2), int(d3)]


x = np.arange(2022, 2025, 1)
k = np.arange(2006, 2025, 1)

p = np.array(data_temp['0 BR'])
p_ = np.concatenate((p, d))


year = range(2006, 2022)
plt.title('year vs pirce')
plt.ylabel('average price')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.xlim(2006, 2024)
plt.plot(data_temp['year'], data_temp['0 BR'])
plt.plot(x, actual_predictions)
for x1, y1 in zip(k, p_):
    print(x1, y1)
    plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
plt.savefig("LSTM_pic.png")
plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


dataset = read_csv('data_dispose3.csv', header=0, index_col=0)
dataset = dataset[['0 BR', 'Burglary', 'interest']]
values = dataset.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))  
scaled = scaler.fit_transform(values)   
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[3, 4]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
n_train_year = 10    
train = values[:n_train_year, :]   
test = values[n_train_year:, :]    
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
print(history)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

raw = inv_y.size
inv_y = inv_y[-3:]
inv_yHat = inv_yhat[-3:]
print(inv_yHat)
pyplot.plot(inv_yHat, label='forecast')  
pyplot.plot(inv_y, label='observation')  
pyplot.ylabel('price')
pyplot.legend()
pyplot.show()

