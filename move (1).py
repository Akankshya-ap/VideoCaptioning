import numpy as np
import shutil
import os

path='features/visual/'
path=os.path.expanduser(path)
filenames=[x for x in os.listdir(path)]
for f in filenames:
    if np.random.rand(1)<0.8:
        shutil.move(path+f,'features/train_features/'+f)


filenames=[x for x in os.listdir(path)]
for f in filenames:
        shutil.move(path+f,'features/test_features/'+f)
