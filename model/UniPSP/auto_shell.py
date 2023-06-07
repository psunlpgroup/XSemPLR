# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('cmd',  type=str)
parser.add_argument('gpu', type=int, default=-1,
                    help='index of the gpu that is waiting for running.')
args = parser.parse_args()

cmd = args.cmd
gpu = args.gpu


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_index = gpu*4
    gpu_memory = int(gpu_status[gpu_index + 2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[gpu_index + 1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 10000:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()