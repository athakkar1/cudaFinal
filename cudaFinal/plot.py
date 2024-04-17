import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('output.csv', names=['Frequency', 'Amplitude'])
real_freq = data['Frequency'] * 44100  / 65536
# Plot the data
plt.plot(real_freq, data['Amplitude'])
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Frequency vs Amplitude')
plt.show()