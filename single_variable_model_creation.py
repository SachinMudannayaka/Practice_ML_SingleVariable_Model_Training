import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Book1.csv")

data

plt.scatter(data.videos, data.views, color = "green")
plt.xlabel("Number of Videos")
plt.ylabel("Total Views")

data.views
data.videos

x=np.array(data.videos.values)
x

y=np.array(data.views.values)
y

model = LinearRegression()
model.fit(x.reshape((-1,1)),y)

new_x = np.array([45]).reshape((-1,1))
new_x

predicted_value = model.predict(new_x)
predicted_value

plt.scatter(data.videos, data.views, color = "green")
plt.xlabel("Number of Videos")
plt.ylabel("Total Views")
m,c= np.polyfit(x,y,1)
plt.plot(x,m*x+c)
