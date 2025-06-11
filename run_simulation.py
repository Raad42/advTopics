import subprocess
import sys
import os

def run_simulation(price, baristas):
    simulator_path = os.path.join('simulator') 
    result = subprocess.run(
        [simulator_path, str(price), str(baristas)],
        capture_output=True,
        text=True
    )
    for line in result.stdout.splitlines():
        if "Profit (captured):" in line:
            return float(line.split(":")[1].strip())
    return None
