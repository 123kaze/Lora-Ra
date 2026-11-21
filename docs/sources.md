# 核实的来源（2026-11-22）

- 方案原文：根目录`研究方案.md`（不改动）。
- Variable-Width Transformers：https://arxiv.org/abs/2606.18246 ，§2说明截断/从最近参与计算的层恢复维度；无历史维度填零。
- 官方代码：https://github.com/ZhaofengWu/variable-width-transformers ，固定commit `69dde8143d9a7912de353b57093d36d8788070d4`。本实现参考机制描述，未复制上游代码，未安装其多GPU训练依赖。
- 权重状态：https://github.com/ZhaofengWu/variable-width-transformers/issues/1 ，读取README、仓库文件树及全部Issue评论，未找到可直接下载的官方权重。该结论不代表对互联网全部第三方checkpoint做过穷尽搜索。
- Tiny Shakespeare：https://github.com/karpathy/char-rnn/tree/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare 。下载保留原始文件与SHA-256；数据不加入代码仓库。
- PyTorch安装：https://pytorch.org/get-started/locally/ ，使用官方CUDA wheel索引；实际精确版本另存run环境记录。
