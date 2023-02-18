import torch
import matplotlib.pyplot as plt

x = torch.tensor([1, 2, 3, 4])
y = x ** 2
fig = plt.figure()


ax = fig.add_subplot(221)
ax.set_title("p1")
ax.set_xlabel("x1")
ax.set_ylabel("y1")
ax.plot(x, y, color="r",marker="o",linestyle="dashed")
ax.axis([0, 10,0,50])


ax1 = fig.add_subplot(222)
ax1.set_title("p2")
ax1.set_xlabel("x2")
ax1.set_ylabel("y2")
ax1.plot(x, y, color="r",marker="o",linestyle="dashed")
ax1.axis([0, 10,0,50])
plt.show()