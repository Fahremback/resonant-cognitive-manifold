import subprocess
import time

def run():
    p = subprocess.Popen(
        [r"build\Release\rcm.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print("Enviando 'large'...")
    p.stdin.write("large\n")
    p.stdin.flush()
    
    # Vamos esperar 5 segundos e ler a saida
    time.sleep(5)
    
    print("Enviando '/exit'...")
    p.stdin.write("/exit\n")
    p.stdin.flush()
    
    stdout, stderr = p.communicate(timeout=5)
    print("--- STDOUT ---")
    print(stdout)
    print("--- STDERR ---")
    print(stderr)

if __name__ == "__main__":
    run()
