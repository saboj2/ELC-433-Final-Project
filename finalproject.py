# use scipy.io to read in .wav files

import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt

#scale down audio file?
fs, d_original = scipy.io.wavfile.read("song1.wav", False)

A = 1
t = numpy.arange(len(d_original))
noise =  A*numpy.sin(60*t)
d_noise = d_original + noise

y = numpy.zeros(len(d_original))
e = numpy.zeros(len(d_original))

filter_order = 1000
w = numpy.ones(filter_order)
lr = 0.5
noise = numpy.pad(noise,(filter_order,filter_order),'constant', constant_values=(0,0))
for n in t:
    for m in range(0, filter_order):
        y[n] = noise[n-m]*w[m]
    e[n] = d_noise[n] - y[n]
    w = w + lr*noise[n]*e[n]


plt.figure(1)
plt.plot(t, d_original)
plt.figure(2)
plt.plot(t, d_noise)
plt.figure(3)
plt.plot(t, y)

plt.show()