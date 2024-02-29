# Embedding训练报告

## 训练设计

由于深度学习并不是非常熟悉，因此在Embedding的自监督训练上遇到了不小的挑战：

* Embedding是什么？
* 实验设计？如何实现自监督训练？ 损失函数怎么构建？
* Transformer的decoder-only version 的每一层是什么，输入输出是什么，维度是多少
* 模型中数据流维度的控制和调整

为了解决上述问题，我们还是先去好好学习了一下相关知识，再完成本次的Embedding学习



## Embedding

`Embedding` 是将离散型的输入（如单词、类别等）映射为连续型的向量的过程

在本次任务中，是将所有的宏观数据进行向量化，例如将一个具体的数值Embedding成一个高维的嵌入向量，嵌入向量更好地捕捉输入数据的信息。

在embedding的生成方面，我们一开始认为是直接可以采用`nn.embedding`模块来进行嵌入操作，但是后面和wdd学长探讨之后，发现这个函数是适用于nlp领域中的，并不适用于当前的时序信息，这里所需要做的就是通过一个`nn.Linear`层将宏观数据通过线性变化生成embedding。

实现方面有两个文件：

* Train_embedding(nnEmbedding).ipynb `不适用于该场景，效果也比nn.Linear差一些`
* Train_embedding(linear).ipynb `下文主要介绍的模型`

## 实验设计

在实验中，在实验设计上我们采用以下步骤：

1. **数据准备：** 将宏观数据按照时间序列划分，需要对数据进行预处理，将数据标准化到一定范围的整数，方面后续的Embedding操作，使用pytorch中的dataloader来生成迭代器。

2. **模型架构：** 使用Transformer架构，去掉decoder中的与encoder交互的注意力部分，留下自回归（autoregressive）结构。这样，模型利用decoder中的mask multihead self-attention结构，可以根据过去的信息预测未来的embedding。

3. **训练目标：** 将模型设计成用前几天的数据的Embedding向量构成的时序信息来预测下一天的embedding，这样的设计就可以更好的确定训练的目标，同时这样也可以证明当前训练的embedding是可以有预测未来信息的能力，可以更好的开展后续的任务。训练也是采用了自监督的学习任务。采用了滑动窗口的方式来构建训练样本，确保训练样本的样本是真实未来embedding。

4. **损失函数：** 构建损失函数来衡量模型学到的embedding和真实数据之间的差异，我们使用MSE损失函数，对比损失的目标是使正样本（真实的未来embedding）在噪声样本中更易于区分。

   

## 模型选择（My-TransformerDecoder）

在模型选择上，我们并没有采用传统的Transformer的decoder层，这是因为传统的Decoder中含有与encoder层结果交互的多头注意力，在我们的模型中并没有encoder层，同时pytorch提供的封装好的decoderLayer并不能调整其内部的结构，因此我们在这里简单实现了一下我们自己的TransformerDecoder层，其中去除了与encoder交互的多头注意力模块：

```python
class MyDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=1):
        super(MyDecoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # 带掩码的自注意力机制
        attn_output, _ = self.self_attention(x, x, x, attn_mask=self.generate_mask(x.size(0)))
        # 残差链接
        x = x + attn_output
        x = self.norm1(x)
        # 前馈神经网络
        ff_output = self.feedforward(x)
        # 残差链接
        x = x + ff_output
        x = self.norm2(x)
        return x

    def generate_mask(self, size):
        # 生成上三角掩码
        mask = torch.triu(torch.ones(size, size), diagonal=1).to(device)
        return mask
```



## 数据的维度

由于在本次编写代码时，所遇到的最大的问题就是张量的维度的控制，因此在这里进行总结：

1. **原始数据（data）：**
   - Shape: (Num_months, Num_indicators)
   - 描述: 包含 46 个指标的 200 个月的时序数据。
2. **数据处理（MyDataset 类）：**
   - 输出数据 Shape: (Batch_size，Seq_len, Num_indicators)
   - 描述: MyDataset 类在 `__getitem__` 方法中返回的是每个样本的输入数据和目标数据。输入数据是过去 Seq_len 个月的时序数据，而目标数据是当前的 embedding。
3. **Embedding层（DataEmbedding 类）：**
   - 输入 Shape: (Batch_size, Seq_len, Num_indicators)
   
   - 输出 Shape: (Batch_size,Seq_len,  Embedding_size)
   
   - （nn.Embedding）在这里将所有宏观向量的Embedding进行了拼接，因此这样每一个月均能得到一个整体的Embedding，因此在后续训练时序数据时，训练效果会更好。
   
      将原始指标数据嵌入到低维度表示的层。每个指标都有一个独立的嵌入，通过 `torch.stack` 连接所有嵌入。
   
     （nn.Linear）在此处生成的总的维度为`Embedding_size`
4. **Transformer Decoder 模型（MyTransformerDecoder 类）：**
   
   - 输入 Shape: ( Batch_size,Seq_len,  Embedding_size) `以nn.linear生成的嵌入维度为准`
   - 输出 Shape: (Batch_size, Embedding_size)
   - 描述: 使用 Transformer 的 decoder-only 结构。Seq_len表示时间序列长度，Batch_size 表示批处理大小，Embedding_size 表示嵌入的总维度。
5. **Loss计算和训练过程：**
   - 损失计算输入 Shape: (Batch_size, Embedding_size)
   - 描述: 在每个训练步骤中，计算模型输出和目标 embedding 之间的损失。损失函数 `nn.MSELoss` 期望输入和目标的形状匹配。

另外，这些维度信息在代码中均有注释



## 总结

Embedding的训练的损失如下：

<img src="D:\desk\量化\20231124-主动量化研究课题\PolicyConfiguration\report\embedding训练\loss.png" alt="loss" style="zoom:80%;" />

<img src="D:\desk\量化\20231124-主动量化研究课题\PolicyConfiguration\report\embedding训练\loss_image.png" alt="loss_image" style="zoom:80%;" />

实验的细节可以在代码文件`Train_embedding(linear).ipynb`中查看

   


