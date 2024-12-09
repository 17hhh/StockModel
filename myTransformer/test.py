# 初始化配置
import qlib
from qlib.constant import REG_CN
data_uri = '~/.qlib/qlib_data/cn_data/'
qlib.init(provider_uri=data_uri, region=REG_CN)

# # 使用"配置"进行实例化
# from qlib.utils import init_instance_by_config
# qdl_config = {
#     "class": "QlibDataLoader",
#     "module_path": "qlib.data.dataset.loader",
#     "kwargs": {
#         "config": {
#             "feature": (['EMA($close, 10)', 'EMA($close, 30)'], ['EMA10', 'EMA30'] ),
#             "label": (['Ref($close, -1)/$close - 1',],['RET_1',]),
#         },
#         "freq": 'day',
#     },
# }
# qdl = init_instance_by_config(qdl_config)
# market = 'csi300' # 沪深300股票池代码，在instruments文件夹下有对应的sh000300.txt
# qdl.load(instruments=market, start_time='20200101', end_time='20200110')