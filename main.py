#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("./datasets/graduation_rate.csv")
df.head(5)
#%%
x_axis = 'parental income'
y_axis = 'years to graduate'
#%%
x = np.array(df[x_axis]).reshape((-1, 1))
y = np.array(df[y_axis])

plt.plot(x, y)

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Linear Regression Raw Data')
plt.show()
# %%
model = LinearRegression().fit(x,y)
# %%
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
# %%
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

diabetes_y_pred = model.predict(x)

# Plot outputs
plt.scatter(x, y, color="black")
plt.plot(x, y_pred, color="blue", linewidth=3)

plt.xlabel(x_axis)
plt.ylabel(y_axis)

plt.show()
# %%
