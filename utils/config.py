import pandas as pd
import os


def args_to_csv(dst_path, config):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    dict_ = vars(config)
    s = pd.Series(data=dict_)
    print('====== Parameters ======')
    print(s)
    print('========================')
    s.to_csv(dst_path)