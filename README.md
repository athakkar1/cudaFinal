Used FIR filter coefficients and an optimized convolution kernel to provide real time audio filtering that takes in mic input in python, pipes it to my cuda kernel, and pipes it back to python to play over my headphones using pyAudio. There are also broken optimized DFT kernels which for some reason do not work in the real-time audio system but do work when they are standalone, and are pretty damn fast at about 200 ms for a 65536 point DFT. Since the DFT didn't work, I tried out some time domain tech by frequency shifting my signal by multiplying a complex exponential in the time domain as per the Fourier Transform properties which worked (kinda). 

CONVOLUTION KERNEL:


https://github.com/athakkar1/cudaFinal/assets/96598825/25df1b1c-07d2-4194-a11a-bdd0d0d09169



https://github.com/athakkar1/cudaFinal/assets/96598825/bc25e2af-fb74-4d62-bcbc-4fc420bf7ce9



https://github.com/athakkar1/cudaFinal/assets/96598825/9fd4009b-4820-4af0-a725-04d11301fc19

DFT KERNEL:


https://github.com/athakkar1/cudaFinal/assets/96598825/3e77ab5b-f0c2-4b1d-b55f-a8c4a68859c6



https://github.com/athakkar1/cudaFinal/assets/96598825/7805e284-2cc2-4d9f-9677-b64a86298f96

