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

