# use scipy.io to read in .wav files

import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import random

fs, d_original = scipy.io.wavfile.read("champions.wav", False)

#cut audio clip for faster processing
#d_original = d_original[1000000:1500000]
#Scale to float between -1 and 1
d_original = d_original/32768.0
#Create time variable length of audio clip
t = numpy.arange(len(d_original))

#Generate Noise
A = 1
noise =  A*numpy.sin(60*t/1000.0)

rand = numpy.zeros(len(t))

#Random filter (convolution)
for k in t:
    rand[k] = float(random.randrange(1,10,1))/1600

noise_rand = numpy.convolve(noise, rand, 'same')

plt.figure(1)
plt.plot(noise)

plt.figure(2)
plt.plot(noise_rand)


p_signal= numpy.sqrt(numpy.mean(d_original**2))
p_noise= numpy.sqrt(numpy.mean(noise**2))
snr_before = p_signal/p_noise
print(snr_before)

#Add interference
d_noise = d_original + noise_rand

#plot the original audio signal and the noisy audio signal
plt.figure(3)
plt.plot(d_original)
plt.title("Original Signal")
plt.xlabel("Sample n")
plt.ylabel("Amplitude s[n]")

plt.figure(4)
plt.plot(d_noise)
plt.title("Desired Signal + Noise")
plt.xlabel("Sample n")
plt.ylabel("Amplitude d[n]")

#Filter parameters
filter_order = 100
lr = 0.0001

#init y and error vectors, and weight vector
y = numpy.zeros(len(d_original))
e = numpy.zeros(len(d_original))
w = numpy.zeros(filter_order)

error = numpy.zeros(len(d_original))
error2 = numpy.zeros(len(d_original))

#zero pad input to filter (noise)
noise = numpy.pad(noise,(filter_order-1,filter_order-1),'constant', constant_values=(0,0))
#Loop through every sample
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
    
    error[n] = d_original[n] - e[n]
    
p_signal_filt= numpy.sqrt(numpy.mean(e**2))
p_noise_filt= numpy.sqrt(numpy.mean(y**2))
snr_after = p_signal_filt/p_noise_filt
print(snr_after)

plt.figure(5)
plt.plot(e)
plt.title("Filtered Signal")
plt.xlabel("Sample n")
plt.ylabel("Amplitude e[n]")

plt.figure(6)
plt.plot(error)
plt.title("Error")
plt.xlabel("Sample n")
plt.ylabel("Amplitude error[n]")

plt.figure(7)
plt.plot(error^2)
plt.title("Error^2")
plt.xlabel("Sample n")
plt.ylabel("Amplitude error[n]")

plt.show()


d_noise = numpy.asarray(d_noise*32768).astype(numpy.int16)
scipy.io.wavfile.write("champions_noise.wav", fs, d_noise)

e = numpy.asarray(e*32768).astype(numpy.int16)
scipy.io.wavfile.write("champions_filter.wav", fs, e)