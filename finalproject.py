# use scipy.io to read in .wav files

import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt

#scale down audio file?
fs, d_original = scipy.io.wavfile.read("champions.wav", False)

d_original = d_original[1000000:1100000]

A = 1
t = numpy.arange(len(d_original))


d_original = d_original/32768.0


noise =  A*numpy.sin(60*t/1000.0)
d_noise = d_original + noise

plt.figure(1)
plt.plot(d_original)

plt.figure(2)
plt.plot(d_noise)


y = numpy.zeros(len(d_original))
e = numpy.zeros(len(d_original))

filter_order = 1000
w = numpy.ones(filter_order)
lr = 0.0001

noise = 5*numpy.pad(noise,(filter_order-1,filter_order-1),'constant', constant_values=(0,0))



for n in t:

    for m in range(0, filter_order):
        #fix for negative indexing
        y[n] += noise[n-m+filter_order-1]*w[m]

    e[n] = d_noise[n] - y[n]

    for j in range(0, filter_order):
        w[j] = w[j] + lr*noise[n-j+filter_order-1]*e[n]


plt.figure(3)
plt.plot(e)
#plt.ylim((-0.1, 0.1))

plt.figure(4)
plt.plot(y)

plt.show()


d_noise = numpy.asarray(d_noise*32768).astype(numpy.int16)
scipy.io.wavfile.write("champions_noise.wav", fs, d_noise)

e = numpy.asarray(e*32768).astype(numpy.int16)
scipy.io.wavfile.write("champions_filter.wav", fs, e)