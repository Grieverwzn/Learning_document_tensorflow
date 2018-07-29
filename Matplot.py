import matplotlib.pyplot as plt
import numpy as np



x =np.linspace(-4,4,50)
y1=3*x+4
y2=x**2

plt.figure(num=1,figsize=(7,6))
plt.plot(x,y1)
plt.plot(x,y2,color="red",linewidth=3.0,linestyle="--")
plt.figure(num=2,figsize=(7,6))
plt.plot(x,y2,color="red",linewidth=3.0,linestyle="--")
plt.show()




