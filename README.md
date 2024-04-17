Used FIR filter coefficients and an optimized convolution kernel to provide real time audio filtering that takes in mic input in python, pipes it to my cuda kernel, and pipes it back to python to play over my headphones using pyAudio. There are also broken optimized DFT kernels which for some reason do not work in the real-time audio system but do work when they are standalone, and are pretty damn fast at about 200 ms for a 65536 point DFT. Since the DFT didn't work, I tried out some time domain tech by frequency shifting my signal by multiplying a complex exponential in the time domain as per the Fourier Transform properties which worked (kinda). 


https://github.com/athakkar1/cudaFinal/assets/96598825/18a7667b-bb9b-4afe-a963-69a369e978e1



https://github.com/athakkar1/cudaFinal/assets/96598825/f063161d-2ff0-4d15-818e-d1cca6af6aea


https://github.com/athakkar1/cudaFinal/assets/96598825/3b3c692e-21b9-401b-a96b-33ed67e715b2

