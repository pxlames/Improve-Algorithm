'''
帮我定义接口，实现一个多态示例。
仿照java的多态实现方式。
类继承父类，实现父类接口，父类调用子类方法。        
'''
from abc import ABC, abstractmethod
from typing import List

class Animal:
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof"

class Cat(Animal):
    def make_sound(self):
        return "Meow"

dog = Dog()
cat = Cat()

print(dog.make_sound())
print(cat.make_sound())