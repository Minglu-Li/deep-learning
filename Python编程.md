# 函数使用

## print函数

方法1：用print() 输入多个信息逗号，分隔的方法，虽然比较山寨但是可用，就是比较繁琐
`print(sh1.name,"这个表有",rows,"行",columns,"列",sep="")`
用逗号 ,把字符串，数字，变量分隔，拼凑成一句话，但是需要把sep设置为""不空格，否则默认为 " " 显示结果比较丑
方法2：print("%s%d" %{var1,var2}) 格式
`print("%s这个表有%d行%d列"  %(sh1,rows,columns))`
s% 字符串
d%  数字
方法3：print("{0}{1}".format(var1,var2)) 格式
`print("{0}这个表有{1}行{2}列".format(sh1.name,rows,columns))`
方法4：print(f"{var1}{var2}") 格式
f也是format，这是一种比较新也是比较简便的格式
`print(f"{sh1.name}这个表有{rows}行{columns}列")`


## Zip函数

在 Python 中，`zip()` 函数是一个非常有用的内置函数，用于将多个可迭代对象（如列表、元组等）“压缩”在一起，形成一个新的迭代器，它可以逐个组合来自每个可迭代对象的元素。

### 1. `zip()` 函数的基本用法

`zip()` 函数的语法如下：

```
zip(*iterables)
```

其中，`iterables` 可以是任意数量的可迭代对象（如列表、元组、字符串等）。

#### 示例

假设有两个列表，我们想要将它们的元素一一对应地组合起来：

```
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

zipped = zip(list1, list2)
print(list(zipped))
```

**输出：**

```
[(1, 'a'), (2, 'b'), (3, 'c')]
```

在这个例子中，`zip()` 函数将 `list1` 和 `list2` 中的元素组合成了 3 个元组，每个元组包含来自两个列表的对应元素。

### 2. `zip()` 函数的特点

- `zip()` 函数返回一个迭代器对象，每次调用会生成一个元组，包含来自所有输入可迭代对象的对应元素。
- 如果输入的多个可迭代对象的长度不相等，`zip()` 会在最短的可迭代对象耗尽时停止生成元组。这意味着它会忽略较长的可迭代对象中多余的元素。

#### 示例：不同长度的可迭代对象

```
list1 = [1, 2, 3, 4]
list2 = ['a', 'b']

zipped = zip(list1, list2)
print(list(zipped))
```

**输出：**

```
[(1, 'a'), (2, 'b')]
```

### 3. `zip()` 的常见用途

- **并行迭代**：`zip()` 经常用于在循环中同时遍历多个序列。

  ```
  names = ['Alice', 'Bob', 'Charlie']
  ages = [25, 30, 35]
  
  for name, age in zip(names, ages):
      print(f"{name} is {age} years old.")
  ```

  **输出：**

  ```
  Alice is 25 years old.
  Bob is 30 years old.
  Charlie is 35 years old.
  ```

- **解压缩**：可以使用 `zip(*zipped_list)` 的形式将压缩后的对象解压缩回原来的形式。

  ```
  pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
  numbers, letters = zip(*pairs)
  print(numbers)  # (1, 2, 3)
  print(letters)  # ('a', 'b', 'c')
  ```

### 4. `zip()` 函数的返回类型

- 在 Python 3 中，`zip()` 返回一个迭代器，这样节省内存，因为它不需要一次性存储所有的配对结果。
- 如果需要所有的配对结果（如存储在列表中），可以使用 `list()` 或 `tuple()` 将其转换为列表或元组。

## reshape函数

用于改变张量的形状，而不改变张量中元素的数量和元素本身的值。它可以将一个张量重新组织成不同的维度排列，只要新形状的元素总数与原张量的元素总数相同即可。

假设原始形状为 (2, 3, 6)：

- batch_size = 2
- 查询个数 = 3
- num_hiddens = 6
- num_heads = 2

执行 `reshape(2, 3, 2, -1)`：

```python
原维度：2 × 3 × 6
新维度：2 × 3 × 2 × 3 （因为 6 / 2 = 3）
```

`-1` 是PyTorch的自动计算功能：

- 总元素数 = batch_size × 查询数 × num_hiddens
- 新维度 = batch_size × 查询数 × num_heads × (num_hiddens/num_heads)
- 所以 `-1` 会自动计算为 `num_hiddens // num_heads`



# 语法使用

## 可变参数

```python
def func(*args, **kwargs):
    print(args)
    print(kwargs)

func(1,2,3,4, x = 8, y = 9)
```

```python
(1, 2, 3, 4)
{'x': 8, 'y': 9}
```

注意这里的1,2,3,4这种参数，会传递给`*args`，得到的是一个元祖，`x = 8, y = 9`这种会传递给`**kwargs`这种会变成字典

## `__call()__`对象调用

```python
class C:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print("CALL " + str(args[0]))

c = C()
c(1,2,3)
```

```python
CALL 1
```

## `@property` 装饰器

这段代码定义了一个名为 `AttentionDecoder` 的类，继承自 `d2l.Decoder`。下面是对代码的详细解释：

```python
class AttentionDecoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weight(self):
        raise NotImplementedError
```

### 1. 类的定义与继承

```python
class AttentionDecoder(d2l.Decoder):
```

- **`AttentionDecoder`**：这是一个自定义的解码器类，通常用于实现带有注意力机制（Attention Mechanism）的解码器部分。
- **`d2l.Decoder`**：`AttentionDecoder` 继承自 `d2l.Decoder`，这意味着它会继承 `d2l.Decoder` 类中的所有属性和方法，并可以在此基础上进行扩展或重写。

### 2. 构造函数 `__init__`

```python
def __init__(self, **kwargs):
    super(AttentionDecoder, self).__init__(**kwargs)
```

- **`__init__` 方法**：这是类的构造函数，用于初始化对象实例。
- **`\**kwargs`**：表示接受任意数量的关键字参数，并将它们传递给父类的构造函数。
- **`super(AttentionDecoder, self).__init__(\**kwargs)`**：调用父类 (`d2l.Decoder`) 的构造函数，确保父类的初始化逻辑得以执行。

### 3. `@property` 装饰器

```python
@property
def attention_weight(self):
    raise NotImplementedError
```

- **`@property`**：这是一个装饰器，用于将一个方法转换为属性访问。使用 `@property` 后，可以通过 `instance.attention_weight` 的方式访问，而无需像调用方法那样使用括号（`instance.attention_weight()`）。

  **示例：**

  ```python
  decoder = AttentionDecoder()
  weight = decoder.attention_weight  # 通过属性访问，而不是方法调用
  ```

### 4. `raise NotImplementedError`

```python
raise NotImplementedError
```

- **`NotImplementedError`**：这是一个内置的异常，用于指示某个方法或功能尚未实现。在基类中定义抽象方法时，经常会使用这个异常，以确保子类必须重写该方法。

  **作用：**

  - **强制子类实现**：当 `AttentionDecoder` 被继承时，子类必须实现 `attention_weight` 属性。如果子类没有实现，当尝试访问 `attention_weight` 时，会抛出 `NotImplementedError`，提醒开发者需要在子类中实现该属性。

  **示例：**

  ```python
  class MyAttentionDecoder(AttentionDecoder):
      @property
      def attention_weight(self):
          return self._compute_attention_weight()
  
      def _compute_attention_weight(self):
          # 实现具体的注意力权重计算逻辑
          return some_weight
  ```

# Pytorch使用

```
torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
```

`torch.repeat_interleave()`是一个PyTorch函数，它会将`valid_lens`中的每个元素沿着指定的维度（`dim=0`）复制`repeats`次。

`valid_lens`可能是一个包含有效长度的向量，形状通常是(batch_size,)或(batch_size, seq_len)。在这里，`repeat_interleave`将会把每个`valid_lens`中的元素复制`num_heads`次。这是因为每个注意力头都需要知道有效长度，而不是只有一个有效长度。

具体来说，如果原来`valid_lens`的形状是`(batch_size,)`，那么经过`repeat_interleave`后，它的形状变为`(batch_size * num_heads,)`。如果原来的形状是`(batch_size, seq_len)`，则复制操作会沿着第一维（即batch维度）进行。

假设`valid_lens`的形状是`(2, 3)`，表示批次大小为2，每个序列有3个有效长度值。如果`num_heads=4`，经过`repeat_interleave`后，`valid_lens`的形状会变成`(8, 3)`，每个批次的有效长度值会被复制4次。
