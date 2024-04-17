import pyaudio
import subprocess
import array
buffer_size = 16384*4
# Open the C binary
c_program = subprocess.Popen('audio.exe', stdin=subprocess.PIPE, stdout=subprocess.PIPE)

p = pyaudio.PyAudio()
input_stream = p.open(format=pyaudio.paInt32, channels=1, rate=44100, input=True, frames_per_buffer=16384*4)
output_stream = p.open(format=pyaudio.paInt32, channels=1, rate=44100, output=True)

while True:
  try:
    # Read data from the microphone
    data_bytes = input_stream.read(buffer_size)

    # Convert the bytes to an array of 16-bit signed integers
    data_int16 = array.array('i', data_bytes)

    # Send a ready signal to the C program
    print("Gonna Print")
    c_program.stdin.write(b'RDY\n')
    c_program.stdin.flush()
    # Send each integer to the C program
    for sample in data_int16:
        c_program.stdin.write(str(sample).encode() + b'\n')
    c_program.stdin.flush()

    # Wait for the ready signal from the C program
    ready_signal = c_program.stdout.readline()
    if ready_signal != b'RDY\r\n':
        print(f"Error: expected ready signal from C program, got {ready_signal}")
        break

    # Read the processed data from the C program
    processed_data = []
    for _ in range(buffer_size):
        line = c_program.stdout.readline()
        processed_data.append(int(line))
    # Play the processed data through the output stream
    #print(processed_data)
    processed_data_bytes = array.array('i', processed_data).tobytes()
    output_stream.write(processed_data_bytes)
  except Exception as e:
    print(f"Error: {e}")
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    p.terminate()
    c_program.terminate()
    break
  except KeyboardInterrupt:
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    p.terminate()
    c_program.terminate()
    print("Interrupted by user")
    break
