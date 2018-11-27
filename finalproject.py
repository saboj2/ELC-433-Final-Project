# use scipy.io to read in .wav files

import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import random

fs, d_original = scipy.io.wavfile.read("champions.wav", False)

#cut audio clip for faster processing
d_original = d_original[1000000:1200000]
#Scale to float between -1 and 1
d_original = d_original/32768.0
#Create time variable length of audio clip
t = numpy.arange(len(d_original))

#Generate Noise
A = 0.25
noise =  A*numpy.sin(60*t/1000.0)

noise_rand = numpy.zeros(len(t))

#Random filter
for k in t:
    noise_rand[k] = noise[k]*(random.randrange(75,125,1)/100)

p_signal= numpy.sqrt(numpy.mean(d_original**2))
p_noise= numpy.sqrt(numpy.mean(noise**2))
snr_before = p_signal/p_noise
print(snr_before)

#Add interference
d_noise = d_original + noise_rand

#plot the original audio signal and the noisy audio signal
plt.figure(1)
plt.plot(d_original)
plt.title("Original Signal")
plt.xlabel("Sample n")
plt.ylabel("Amplitude s[n]")

plt.figure(2)
plt.plot(d_noise)
plt.title("Desired Signal + Noise")
plt.xlabel("Sample n")
plt.ylabel("Amplitude d[n]")

#Filter parameters
filter_order = 1000
lr = 0.01

#init y and error vectors, and weight vector
y = numpy.zeros(len(d_original))
e = numpy.zeros(len(d_original))
w = numpy.zeros(filter_order)


#zero pad input to filter (noise)
noise = numpy.pad(noise,(filter_order-1,filter_order-1),'constant', constant_values=(0,0))

for something in range(1):
    #Loop through every sample
    print(something)
    for n in t:

        #apply filter with m taps
        for m in range(0, filter_order):

            #Add filter_order - 1 to account for zero padding
            y[n] += noise[n-m+filter_order-1]*w[m]

        #Find error
        e[n] = d_noise[n] - y[n]

        #Weight update
        for j in range(0, filter_order):
            w[j] = w[j] + lr*noise[n-j+filter_order-1]*e[n]

p_signal_filt= numpy.sqrt(numpy.mean(e**2))
p_noise_filt= numpy.sqrt(numpy.mean(y**2))
snr_after = p_signal_filt/p_noise_filt
print(snr_after)

plt.figure(3)
plt.plot(e)
plt.title("Filtered Signal")
plt.xlabel("Sample n")
plt.ylabel("Amplitude e[n]")
#plt.ylim((-0.1, 0.1))

plt.show()


d_noise = numpy.asarray(d_noise*32768).astype(numpy.int16)
scipy.io.wavfile.write("champions_noise.wav", fs, d_noise)

e = numpy.asarray(e*32768).astype(numpy.int16)
scipy.io.wavfile.write("champions_filter.wav", fs, e)