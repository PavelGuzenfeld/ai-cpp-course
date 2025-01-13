import my_module

obj = my_module.MyClass()
print(obj.get_value())  # should print 0

obj.set_value(42)
print(obj.get_value())  # should print 42

# this will throw an exception
try:
    obj.set_value(-1)
except ValueError as e:
    print("oops:", e)
