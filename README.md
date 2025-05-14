# NLP Backdoor Attack and Defense Benchmark

## 项目简介

本项目旨在提供一个全面的基准测试，用于评估和比较自然语言处理（NLP）模型中的后门攻击和防御方法。通过整合多种主流的后门攻击和防御技术，我们希望为研究人员和开发者提供一个标准化的测试平台，以便更好地理解和改进NLP模型的安全性。

## 项目结构

```plaintext
- 攻击方法：本项目支持多种主流的NLP后门攻击方法，包括但不限于：
  - BadNets：通过随机插入固定罕见字符触发器（如cf、mn、bb、tq等）的组合来毒化训练数据。
  - AddSent：通过随机插入特定句子（如“我看了这部电影”）来毒化训练数据。
  - Hidden Killer：通过将输入转化为预定义的语法结构来毒化训练数据。
  - Stylebkd：通过风格迁移模型将输入转化为特定风格（如圣经风格等）来毒化训练数据。

- 防御方法：本项目集成了多种主流的NLP后门防御技术，包括但不限于：
  - ONION：通过删除可疑词后计算句子困惑度变化识别触发器。
  - STRIP：利用后门样本对输入扰动的敏感性差异。叠加随机噪声生成扰动样本，若模型输出熵极低则判定为后门输入。
  - CUBE：使用密度聚类HDBSCAN分离后门和干净样本。
  - BKI：分析LSTM模型中单词对输出的影响，识别并移除包含后门触发词的中毒样本。
  - MuScleLoRA：通过在频率空间中部署多个径向缩放，并对目标模型进行低秩自适应，在更新参数时进一步对齐梯度，从而减轻后门样本的影响。
```

# 项目贡献
本项目引用了以下仓库，为检索后门模型提供了完备的攻击和防御代码：

- Positive Feedback：[git@github.com:RetrievalBackdoorDefense/PositiveFeedback.git](https://github.com/RetrievalBackdoorDefense/PositiveFeedback)
  - 描述：Positive Feedback 提供了一种动态识别高损失样本并执行反向梯度更新的方法，逐步提升中毒样本识别能力，从而有效净化训练过程并防御检索模型中的后门攻击。