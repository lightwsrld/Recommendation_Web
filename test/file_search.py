import pandas
import os
dir = "데이터/포스터"
files = os.listdir(dir)
matching = [s for s in files if "벤허" in s] 
print(matching[0])