import serial
import time

SERIAL_PORT = "/dev/ttyACM0" # Change this to your port
BAUD_RATE = 115200

# --- CONFIGURATION ---
# 1.8 is the motor's full step angle (most common)
# 16 is the microstep setting (1/16 is a common default)
# Change '16' if your jumpers are set differently.
DEGREES_PER_FULL_STEP = 1.8
MICROSTEP_SETTING = 16

try:
    stm_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    time.sleep(2) 

    while True:
        command = input("Enter DEGREES to move (e.g., 90 or -360): ")
        
        if command.lower() == 'exit':
            break
            
        try:
            degrees_to_move = float(command)
            
            # --- THE CONVERSION ---
            steps_to_move = int((degrees_to_move * MICROSTEP_SETTING) / DEGREES_PER_FULL_STEP)

            # Send the calculated STEPS command to the STM32
            stm_serial.write(f"{steps_to_move}\n".encode())
            print(f"Sent: {command} degrees ({steps_to_move} steps)")
            
        except ValueError:
            print("Invalid input. Please enter a number.")

    stm_serial.close()
    print("Connection closed.")

except serial.SerialException as e:
    print(f"Error: Could not open port {SERIAL_PORT}.")
