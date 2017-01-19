import matplotlib.pyplot as plt
import time

plt.ion()

for i in range(10):
    plt.figure(1)
    plt.plot(i, i, '.')
    plt.pause(0.05)
    time.sleep(.5)
    print i
