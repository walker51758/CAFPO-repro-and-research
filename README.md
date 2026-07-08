**论文思路**：将94个公司特征输入条件自编码器（Conditional Autoencoder），压缩提取出 K 个隐含因子（Latent Factors）；这些时序因子随后被送入 LSTM 捕捉动态依赖关系；最后由 PPO / DDPG 等强化学习算法根据 LSTM 输出的状态输出股票权重。

论文里输入的是Gu, Kelly & Xiu (2019), Empirical Asset Pricing via Machine Learning 中的94个Characteristics。当前这套面板已经补上了 `baspread`，所以可用特征数是83个。

样本从2007年1月开始，A 股 2007-01-01 起执行新会计准则，很多财务比率从这时起可比性更好。

**条件自编码器的设计动机**：传统上 β 由时序上的股票收益率与因子组合收益率回归得到，但作者认为 β 应该由公司特征决定。因此模型被拆成两个并行的网络——左侧网络以 94 个公司特征为输入，经神经网络后输出 β；右侧网络以股票收益为输入，经线性层直接产出 K 个隐含因子 f（不再以特征为输入）。两边输出相乘得到预测收益 r̂ = β'f，训练时优化的目标是最小化 min(r − β'f)²，梯度下降会同时调整 β 网络与因子网络，让两者在最小化重构误差的过程中协同学习。

**2026-07-07 CAE修正**：`cafpo_reproduction.train_cae` 现在默认使用 `cae` 已训练并验证过的 paper-faithful latent factors，直接从 `cae/outputs/paper_faithful/paper_cae_CA1_K5_test_YYYY.npz` 读取 `(T, 5)` 因子并按 CAFPO 的 `arrays.dates` 对齐，避免在 CAFPO 中重复训练 autoencoder。需要重新训练时可显式使用 `train_cae(..., cae_mode="paper")`；旧的简化版本仍可通过 `train_cae(..., cae_mode="legacy")` 调用，便于对照实验。

**论文测试的强化学习组合**：DRL 算法共 2 种（PPO、DDPG），奖励函数共 3 种（Log Return、Differential Sharpe Ratio、Differential Downside Deviation Ratio），两两交叉共 6 种组合，目前复现了 PPO + Log Return 这一组。

**第一次测试**，在`cafpo\cafpo_ppo_log_reproduction.ipynb`里面，犯了一个错误。EW（等权）和VW（市值加权）组合不是通常意义上的，它是做多历史收益率靠前的、做空历史收益率靠后的，在多空组合内等权或市值加权。导致cafpo策略看起来打败了EW和VW。

多空组合导致收益率异常低，违背市场规律，所以我**加了一个softmax映射，把权重变成long only**，在`cafpo\cafpo_ppo_log_top200_softmax_longonly.ipynb`和`cafpo\cafpo_ppo_log_ff6_fixed180_experiments.ipynb`里，都有体现。

**修改之后，cafpo的收益率有大幅度的提升**，来到了CR 50%到60%的水平。但排名从多空组合里的第二名变成了最后一名。通过调整softmax的temperature参数（不断增大），发现收益率有所提高，不断靠近EW的水平。因为temperature越高，映射后的权重越平均，所以这反而说明cafpo策略本身起的是副作用。

**我们做了一系列反思和实验**。

1. 考虑到同一个动作维度在不同月份代表不同股票这个问题，我们在每一个split里把股票固定下来，固定为test year期初的180支股票，股票从Fama-French市值和价值因子划分成的六个组合中平均产生。
2. 特征工程，在实验一的基础上做。把那些训练期内180只股票横截面的空值率在50%以上的时间超过半数的特征剔除。每次训练期都做这个筛选.
3. 实验二的基础上，把动作空间缩小到18维，也就是只生产18只股票的权重，但特征数据在输入CAE时仍然是180只股票的数据（为了增加因子在横截面上的信息）。

结果不尽人意：

| 实验 | 方法 | 累计收益 | Sharpe | 最大回撤 |
| --- | --- | --- | --- | --- |
| exp1（180，全特征） | CAFPO_PPO  | +59.2%  | +0.106 | -30.6% |
| exp1                | EqualWeight| +68.0%  | +0.112 | -29.5% |
| exp1                | ValueWeight| +63.2%  | +0.117 | -25.7% |
| exp2（180，特征过滤）| CAFPO_PPO | +47.5%  | +0.093 | -31.0% |
| exp2                | EqualWeight| +68.0%  | +0.112 | -29.5% |
| exp2                | ValueWeight| +63.2%  | +0.117 | -25.7% |
| exp3（180→18）      | CAFPO_PPO | +4.7%   | +0.051 | -62.5% |
| exp3                | EqualWeight| +91.6%  | +0.119 | -35.5% |
| exp3                | ValueWeight| +139.8% | +0.152 | -33.7% |

**固定之后，cafpo并没有好转**，不论是从排名上还是绝对累计收益上（如果是原本的top200逻辑，CR是66%。不过这个参考性不强，因为股票都不一样）。**exp2有轻微的负效果**。似乎注意到实际被特征工程剔除的特征只有一个，是real_estate，这说明两件事，特征工程多此一举（你只能筛掉一个特征，没啥用），以及，虽然有一些特征数量不那么齐整，但它的信息含量是挺大的，不好随便删。最令人沮丧的莫过于exp3，原本以为让它学180个action太多了，想着缩小action能够促进更有效的学习，结果完全崩溃了。我猜想，这可能暴露了一个大问题。180个action，可能持仓还比较分散；18个action，持仓就变得集中了。**cafpo在里面起的是负作用，它没法选出合适的资产，所有的收益提升都来自与EW策略相似度的提升。**这和我前面的分析相呼应。

我还尝试了一个非常有意思的做法，**把action改成6维的FF6 group allocation**，每一个group里有176支股票。这么做的目的是，在保证CAE输入的横截面信息充分的情况下，尽量缩小动作维度。这个想法的缘起是，我有一点怀疑exp2里输入180个股票的特征得到因子和输出18个股票的权重会出现错配（其实我从理论分析上觉得不会有这个问题，因为前面一步用于产生描述市场状态的因子，后面一步用于构建组合，是两个不同的事情，不过万一呢？），于是我就想着让输入和输出的股票数匹配。

| 方法 | 累计收益 | Sharpe | 最大回撤 |
| --- | --- | --- | --- |
| FF6GroupEqualWeight | +53.1% | 0.373 | -26.3% |
| FF6GroupValueWeight | +48.3% | 0.381 | -23.5% |
| CAFPO T=0.2         | +17.1% | 0.196 | -47.1% |

## Uncommitted Data Files

以下文件未提交到 git 仓库（由 `.gitignore` 排除），需要从上游数据源重新获取或由脚本生成：

| 路径 | 说明 |
|------|------|
| `rqdata_output/` | 通过RQData接口下载的原始数据（K线、行情、财务因子等），由下载脚本生成 |
| `_smoke_*.py` | 快速冒烟测试脚本，仅供本地调试用 |
| `*.planA_backup` / `*.fixedstart_backup` | 实验过程中的手动备份文件 |
| `notebook_run.pid` / `notebook_run_*.log` | Jupyter notebook 后台运行产生的临时文件 |

以下文件已提交：

| 路径 | 说明 |
|------|------|
| `Gu_Kelly_Xiu_94_Firm_Characteristics.xlsx` | 94个公司特征数据（论文核心输入） |
| `download_rqdata_direct_easy.py` / `download_rqdata_strict45.py` | RQData 数据下载脚本 |
| `prepare_cafpo_reproduction_panel.py` | 面板数据预处理脚本 |
| `cafpo_reproduction.py` / `cafpo_ff6_experiments.py` 等 | 核心实验代码 |

