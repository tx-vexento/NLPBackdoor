import os
import shutil


class PrintColor:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def copy_shell(shell_path, output_dir):
    target_file_path = os.path.join(output_dir, "run.sh")
    try:
        shutil.copy(shell_path, target_file_path)
        print(f"文件已成功复制到 {target_file_path}")
    except IOError as e:
        print(f"无法复制文件. {e}")
