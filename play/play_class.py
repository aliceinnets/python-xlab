'''
Created on 1 Apr 2017

@author: alice<aliceinnets[at]gmail.com>
'''

class Employee:
    'Common base class for all employees'
    empCount = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1
    
    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)
        
    def displayEmployee(self):
        print("Name: ", self.name, ", Salary: ", self.salary)
    
class Parent:        # define parent class
    parentAttr = 100
    def __init__(self):
        print("Calling parent constructor")
    
    def parentMethod(self):
        print('Calling parent method')
        
    def setAttr(self, attr):
        Parent.parentAttr = attr
    
    def getAttr(self):
        print("Parent attribute :", Parent.parentAttr)

class Child(Parent): # define child class
    def __init__(self):
        print("Calling child constructor")
    
    def childMethod(self):
        print('Calling child method')


def main():
    emp1 = Employee("Zara", 2000)
    emp2 = Employee("Manni", 5000)
    
    emp1.displayEmployee()
    emp2.displayEmployee()
    print("Total Employee %d" % Employee.empCount)
    
    emp1.age = 7
    emp2.age = 8
#     del emp1.age
    
    hasattr(emp1, 'age')
    getattr(emp1, 'age')
    setattr(emp1, 'age', 8)
    delattr(emp1, 'age')
    
    print("Employee.__doc__:", Employee.__doc__)
    print("Employee.__name__:", Employee.__name__)
    print("Employee.__module__:", Employee.__module__)
    print("Employee.__bases__:", Employee.__bases__)
    print("Employee.__dict__:", Employee.__dict__)
    
    c = Child()          # instance of child
    c.childMethod()      # child calls its method
    c.parentMethod()     # calls parent's method
    c.setAttr(200)       # again call parent's method
    c.getAttr()          # again call parent's method
    
    
if __name__ == '__main__':
    main()