# evaluate_para_eqa.py

"""
1. 在run_para_eqa.py运行完成后，根据results和logs来统计EQA信息
2. 统计questions_init正确率、questions_follow_up正确率、总正确率
3. 统计每一组问题中每一个问题的的（规范化）步数、平均步数、最大步数、最小步数
4. 针对数据集中的所有问题组，计算总体正确率、步数等指标
5. 可视化上述统计结果
6. 输出统计结果到一个新的文件中
"""
