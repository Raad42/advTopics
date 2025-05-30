import subprocess
import sys

def run_simulation(price, baristas):
    result = subprocess.run(
        ['./simulator', str(price), str(baristas)],
        capture_output=True,
        text=True
    )
    for line in result.stdout.splitlines():
        if "Profit (captured):" in line:
            return float(line.split(":")[1].strip())
    return None
