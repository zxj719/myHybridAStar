### 1. 装饰器

#### 基本概念
- **装饰器**是一种高级函数，它接收一个函数或类作为参数并返回一个新的函数或类。装饰器用于在不修改原函数或类定义的情况下，动态地增加或修改其功能。
- **定义装饰器**：
  ```python
  def my_decorator(func):
      def wrapper():
          print("Something is happening before the function is called.")
          func()
          print("Something is happening after the function is called.")
      return wrapper
  ```
- **使用装饰器**：
  ```python
  @my_decorator
  def say_hello():
      print("Hello!")
  
  say_hello()
  ```

#### 带参数的装饰器
- **定义带参数的装饰器**：
  ```python
  def my_decorator(func):
      def wrapper(*args, **kwargs):
          print("Something is happening before the function is called.")
          result = func(*args, **kwargs)
          print("Something is happening after the function is called.")
          return result
      return wrapper
  ```

#### 类装饰器
- **定义类装饰器**：
  ```python
  def add_method(cls):
      cls.new_method = lambda self: f"New method in {self.__class__.__name__}"
      return cls

  @add_method
  class MyClass:
      def __init__(self, value):
          self.value = value
  
  # 使用新添加的方法
  obj = MyClass(10)
  print(obj.new_method())  # 输出: New method in MyClass
  ```

### 2. 类属性和实例属性

#### 基本概念
- **类属性**：属于整个类的变量，由所有实例共享。
- **实例属性**：属于实例的变量，每个实例都有自己独立的一份。
- **定义类属性和实例属性**：
  ```python
  class MyClass:
      class_attr = "This is a class attribute"

      def __init__(self, value):
          self.instance_attr = value
  
      def show_attrs(self):
          print(f"Class attribute: {MyClass.class_attr}")
          print(f"Instance attribute: {self.instance_attr}")

  # 创建实例并访问属性
  obj1 = MyClass(10)
  obj2 = MyClass(20)
  obj1.show_attrs()
  obj2.show_attrs()
  ```

#### 类方法和静态方法
- **类方法**：使用 `@classmethod` 装饰，第一个参数是类对象 `cls`。
  ```python
  class MyClass:
      class_attr = "Class Attribute"

      @classmethod
      def class_method(cls):
          return cls.class_attr

  print(MyClass.class_method())  # 输出: Class Attribute
  ```
- **静态方法**：使用 `@staticmethod` 装饰，不需要 `self` 或 `cls` 参数。
  ```python
  class MyClass:
      @staticmethod
      def static_method():
          print("This is a static method.")

  MyClass.static_method()  # 输出: This is a static method.
  ```

#### 类方法修改类属性
- **示例**：
  ```python
  class MyClass:
      class_attr = "Class Attribute"

      @classmethod
      def set_class_attr(cls, value):
          cls.class_attr = value

  MyClass.set_class_attr("New Class Attribute")
  print(MyClass.class_attr)  # 输出: New Class Attribute
  ```

### 3. 特殊方法

#### `__init__` 和 `__call__`
- **`__init__`**：初始化方法，用于设置对象的初始状态。
  ```python
  class MyClass:
      def __init__(self, value):
          self.value = value
  ```
- **`__call__`**：使对象实例可以像函数一样被调用。
  ```python
  class CallableClass:
      def __init__(self, value):
          self.value = value

      def __call__(self, increment):
          self.value += increment
          print(f"Value after increment: {self.value}")

  obj = CallableClass(10)
  obj(5)  # 输出: Value after increment: 15
  ```

#### `*args` 和 `**kwargs`
- **`*args`**：接收任意数量的位置参数，并将其作为元组传递。
  ```python
  def func(*args):
      print(args)

  func(1, 2, 3)  # 输出: (1, 2, 3)
  ```
- **`**kwargs`**：接收任意数量的关键字参数，并将其作为字典传递。
  ```python
  def func(**kwargs):
      print(kwargs)

  func(a=1, b=2)  # 输出: {'a': 1, 'b': 2}
  ```