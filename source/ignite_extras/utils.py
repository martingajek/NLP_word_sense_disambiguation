import random, string
import os


def gen_random_str(N=10):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

def get_new_run_directory(_logpath,randomstr=''):
    """ Given log path returns new run directory path """
    runid = lambda x: x.split('_')[0].split('run')[1]
    
    if randomstr:
        randomstr = '_'+randomstr
    if os.path.exists(_logpath):
        _listdirs = [int(runid(d)) for d in os.listdir(_logpath) if  str.startswith(d,'run')]
        _listdirs.sort()
        num = 0
        if _listdirs:
            num = _listdirs[-1]
        _new_run_directory = os.path.join(_logpath,'run{}{}'.format(num+1,randomstr))
    else:
        _new_run_directory = os.path.join(_logpath,'run0{}'.format(randomstr))
        os.makedirs(_new_run_directory)
    return _new_run_directory