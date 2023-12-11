import sys
import os

project_folders = ['utils', 'cli', 'clients', 'config', 'data', 'lobby', 'logs', 'presentation', 'security']
base_path = 'C:/Users/manso/OneDrive/Desktop/python project/'

for folder in project_folders:
    path_to_append = os.path.join(base_path, folder)
    #print(f"Adding to sys.path: {path_to_append}")
    sys.path.append(path_to_append)

try:
    from error import *
    from log_system import *
    from system_info import *
    from cli import *
    from cli_tools import *
    from command import *
    from event import *
    from input_sys import *
    from postgresql import *
    from config import *
    from env import *
    from pickel import *
    from encryption import *
    from setting import *

except ModuleNotFoundError as e:
    print(e)
    #print("Current sys.path:")
    #print(sys.path)
