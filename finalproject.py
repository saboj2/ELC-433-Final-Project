# use scipy.io to read in .wav files

import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt

#scale down audio file?
fs, d_original = scipy.io.wavfile.read("championsShort.wav", False)

d_original = d_original[1000000:1010000]

A = 100
t = numpy.arange(len(d_original))


d_original = d_original/32768.0


noise =  A*numpy.sin(60*t/1000)
d_noise = d_original + noise

y = numpy.zeros(len(d_original))
e = numpy.zeros(len(d_original))

filter_order = 100
w = numpy.zeros(filter_order)
lr = 0.00000001

noise = 5*numpy.pad(noise,(filter_order-1,filter_order-1),'constant', constant_values=(0,0))



for n in t:

    for m in range(0, filter_order):
        #fix for negative indexing
        y[n] += noise[n-m+filter_order-1]*w[m]

    e[n] = d_noise[n] - y[n]

    for j in range(0, filter_order):
        w[j] = w[j] + lr*noise[n-j]*e[n]

plt.figure(1)
plt.plot(t, d_original)
plt.figure(2)
plt.plot(t, d_noise)
plt.figure(3)
plt.plot(e)

plt.show()