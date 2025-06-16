# Paper

> Attention is All you Need

<img src="./assets/image-20250421103149169.png" alt="image-20250421103149169" style="zoom: 80%;" />

<img src="./assets/image-20250421103240727.png" alt="image-20250421103240727" style="zoom:80%;" />

<img src="./assets/image-20250421103408779.png" alt="image-20250421103408779" style="zoom:67%;" />

# 参考文档

https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention

# 模型结构

<img src="./assets/image-20250421103438631.png" alt="image-20250421103438631" style="zoom:80%;" />

<img src="./assets/image-20250421162755610.png" alt="image-20250421162755610" style="zoom:80%;" />

<img src="./assets/image-20250421173339169.png" alt="image-20250421173339169" style="zoom:80%;" />

# 注意力

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

>  除以$\sqrt{d_k}$为什么？
>
> <img src="./assets/image-20250421174535267.png" alt="image-20250421174535267" style="zoom:80%;" />

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # 在不重要的位置上概率赋值为0，能量无穷小
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

