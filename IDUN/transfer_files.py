from paramiko import SSHClient
from scp import SCPClient

REMOTE_HOST = "idun.hpc.ntnu.no"
USERNAME = "askis"
PASSWORD = "OddBrochMannsVeg165!"
REMOTE_PATH = "project"

FILES_TO_TRANSFER = [
    "test.py",
    "train.slurm",
    "../config.py",
    "../data_preparation.py",
    "../model.py",
    "../trainer.py",
    "../main.py",
    "../utils.py",
    "../config.yaml",
    "../label_images.py",
    "../label_ordering.py",
    "../image_labels",
    "../output",
    "../pyproject.toml",
]

def transfer_files_scp(files, remote_host, username, password, remote_path):
    try:
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(remote_host, username=username, password=password)

        with SCPClient(ssh.get_transport()) as scp:
            for file in files:
                print(f"Transferring {file} to {remote_path}")
                scp.put(file, remote_path, recursive=True)

        print("All files transferred successfully.")

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")

# Call the function
transfer_files_scp(FILES_TO_TRANSFER, REMOTE_HOST, USERNAME, PASSWORD, REMOTE_PATH)