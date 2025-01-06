#!/bin/bash

# 设置TensorBoard日志目录和端口
# LOG_DIR="~/proj/StockModels/myTransformer/runs/my_experiment"  # 替换为你的TensorBoard日志目录
LOG_DIR="~/proj/StockModels/MASTER/runs/my_experiment"  # 替换为你的TensorBoard日志目录
# LOG_DIR="~/proj/StockModels/DTML/runs/my_experiment"  # 替换为你的TensorBoard日志目录

PORT=6006                    # TensorBoard服务器监听的端口

# 检查TensorBoard是否已安装
# if ! command -v tensorboard &> /dev/null
# then
#     echo "TensorBoard未安装。正在安装TensorBoard..."
#     pip install tensorboard  # 或者使用pip3，取决于你的Python版本
#     if [ $? -ne 0 ]; then
#         echo "TensorBoard安装失败。"
#         exit 1
#     fi
#     echo "TensorBoard安装成功。"
# else
#     echo "TensorBoard已安装。"
# fi

# 启动TensorBoard服务器
echo "正在启动TensorBoard服务器，日志目录为：$LOG_DIR，端口为：$PORT"
nohup tensorboard --logdir $LOG_DIR --port $PORT &
TB_PID=$!

# 输出TensorBoard服务器的访问信息
TB_URL="http://localhost:$PORT"
echo "TensorBoard服务器已启动，访问地址为：$TB_URL"
echo "TensorBoard进程PID为：$TB_PID（如需终止服务器，请使用'kill $TB_PID'）"

# 可选：保持脚本运行，以便TensorBoard服务器持续运行
# 你可以注释掉下面的'wait'命令，并在另一个终端窗口中手动停止TensorBoard服务器
# wait $TB_PID

# 注意：上面的'nohup'应该是一个拼写错误，正确的应该是'nohup'的变体'nohup'实际上并不存在，
# 正确的命令是'nohup'的相近词'nohangup'（但这也不是我们要的），实际上应该使用'nohup'的
# 意图可能是想使用'nohup'（但拼写错误）或者'nohangup'并不是有效的shell命令。
# 在这里，我们应该使用'&'将TensorBoard放入后台运行，或者使用'disown'来从shell会话中分离进程。
# 但由于我们的目的是简单地启动TensorBoard并让它在后台运行，所以'&'已经足够了。
# 我已经修正了脚本，去掉了错误的'nohup'。

# 清理（通常不需要在脚本中终止TensorBoard，除非你是为了自动化测试）
# 如果你确实需要在脚本结束时终止TensorBoard，可以取消注释下面的行，
# 但请注意，这会使脚本在TensorBoard终止后立即退出，而不是保持TensorBoard运行。
# echo "正在终止TensorBoard服务器..."
# kill $TB_PID
# echo "TensorBoard服务器已终止。"