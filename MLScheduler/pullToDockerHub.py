import subprocess
import time

def execute_command(command):
    while True:
        process = subprocess.Popen(command, shell=True)
        process.wait()
        if process.returncode == 0:
            break
        time.sleep(10)  # 等待十秒后重新执行命令

command = "bash pullToDockerHub.sh"  # 更改这里的命令为你要执行的命令
execute_command(command)

