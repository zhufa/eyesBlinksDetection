# -*- coding: utf-8 -*-

# 曲线拟合

import matplotlib.pyplot as plt
import numpy as np

y = [0.22309655307629433, 0.16298849136246618, 0.19227871415017908, 0.16297582655889586, 0.18507187522501553,
     0.16106665792670294, 0.15430053215971723, 0.15487257090151027, 0.15994159691728851, 0.16630057214607502,
     0.19865732863970464, 0.19222635905523305, 0.2008913597505369, 0.25384105865990381, 0.27366718565898385,
     0.30055763752054754, 0.29882783452175343]
x = range(1, len(y)+1)

z1 = np.polyfit(x, y, 6)  # 用n次多项式拟合
p1 = np.poly1d(z1)
yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
print(yvals)
plot1 = plt.plot(x, y, 'k.', markersize=16, label='$original values$')
plot2 = plt.plot(x, yvals, 'r', lw=3, label='$polyfit values$')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()
# plt.plot(x,y)
plt.pause(100)
