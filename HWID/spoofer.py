import wmi

# Connect to WMI
c = wmi.WMI()

# Ask for CPU info
for processor in c.Win32_Processor():
    print(f"CPU: {processor.Name}")
    print(f"Processor ID: {processor.ProcessorId}")

# Ask for motherboard info
for board in c.Win32_BaseBoard():
    print(f"Motherboard Serial: {board.SerialNumber}")