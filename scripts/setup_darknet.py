import os
import re
import subprocess
import platform
import argparse
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_command(command):
    try:
        subprocess.run(
            [command, "--help"], shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"[+] `{command}` checked.")
    except subprocess.CalledProcessError:
        print(f"[-] `{command}` not installed.")
        print(f"[*] Exiting ...")
        exit(1)


def generate_soln(src_dir, build_dir, config):
    _config = {
        "ENABLE_CUDA": config.get("ENABLE_CUDA", "OFF"),
        "ENABLE_CUDNN": config.get("ENABLE_CUDNN", "OFF"),
        "ENABLE_OPENCV": config.get("DENABLE_OPENCV", "OFF"),
        "ENABLE_CUDA_OPENGL_INTEGRATION": "OFF",
        "ENABLE_OPENCV": "OFF",
    }

    try:
        args = ["cmake", "-G", config.get("vs_version"), "-S", src_dir,"-B", os.path.join(dir, "build")]
        args += [f"-D{attr}={value}" for attr, value in _config.items()]
        subprocess.run(*args, shell=True)
        print("[+] Darknet.sln generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate darknet.sln with GPU support.{e}")
        exit(1)


def build_soln(build_dir):
    try:
        subprocess.run(["msbuild", build_dir, "/target:ALL_BUILD", "/p:Configuration=Release", "/p:Platform=x64"], shell=True)
        print("[+] Darknet built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build darknet.{e}")
        exit(1)



def install_win(dir, use_gpu=False, vs_version="Visual Studio 17 2022"):
    
    print("[*] Checking cmake ...")
    check_command("cmake")
    
    print("[*] Checking msbuild ...")
    check_command("msbuild")

    if use_gpu:
        print("[*] Generating darknet.sln with GPU support.")
        generate_soln(dir, os.path.join(dir, "build"), {'vs_version': vs_version})
        
    else:
        print("[*] Generating darknet.sln with GPU support.")
        generate_soln(dir, os.path.join(dir, "build"), {'vs_version': vs_version})

def install_linux(src_dir, use_gpu = False):
    with open(os.path.join(src_dir, "Makefile"), 'r') as file:
            content = file.read()
    if use_gpu:
        content = re.sub(r"OPENCV=0", "OPENCV=1", content)
        content = re.sub(r"GPU=0", "GPU=1", content)
        content = re.sub(r"CUDNN=0", "CUDNN=1", content)

    content = re.sub(r"LIBSO=0", "LIBSO=1", content)
    with open(os.path.join(src_dir, "Makefile"), 'w') as file:
            file.write(content)
    try:
        subprocess.run(["make"], shell=True)
        print("[+] Darknet built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build darknet.{e}")
        exit(1)
        

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Module for installing darknet")
    parser.add_argument("--gpu", action="store_true", help="If gpu is available.")
    args = parser.parse_args()

    
    # Cleaning pre setup dark-net
    print("[*] Cleaning pre setup dark-net.")
    shutil.rmtree(os.path.join(BASE_DIR, ".darknet"), ignore_errors=False)

    print(f"[*] Cloning dark-net from `https://github.com/AlexeyAB/darknet`.")
    subprocess.run(["git", "clone", "https://github.com/AlexeyAB/darknet", '.darknet'])

    install_linux(os.path.join(BASE_DIR, ".darknet"), use_gpu=args.gpu)
    exit()
    if platform.system() == "Windows":
        print("[*] Installing on windows", args.gpu)
        install_win(os.path.join(BASE_DIR, ".darknet"), use_gpu=args.gpu)
    elif platform.system() == "Linux":
        install_linux(os.path.join(BASE_DIR, ".darknet"), use_gpu=args.gpu)
