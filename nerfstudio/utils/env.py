import os

def set_env_variables():
    """Set some environment variables"""
    if not os.environ.get("DONT_AUTO_DETECT_GPU", False):
        import subprocess
        mingpu_cli = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t ',' -k2 -n | head -n 1 | cut -d ',' -f 1 | xargs"
        mingpu = subprocess.check_output(mingpu_cli, shell=True).decode('utf-8').strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = mingpu
    if not os.environ.get("DONT_REDUCE_THREAD_NUM", False):
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"