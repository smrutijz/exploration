import argparse
import logging

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def devide(x, y):
    return x / y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("number1", type = int, help = "add first number")
    parser.add_argument("number2", type = int, help = "add first number")
    args = parser.parse_args()

    n1 = args.number1
    n2 = args.number2
    
    add_result = add(n1, n2)
    print("Add: {} + {} = {}".format(n1, n2, add_result))

    subtract_result = subtract(n1, n2)
    print("subtract: {} - {} = {}".format(n1, n2, subtract_result))

    multiply_result = multiply(n1, n2)
    print("multiply: {} * {} = {}".format(n1, n2, multiply_result))

    devide_result = devide(n1, n2)
    print("devide: {} / {} = {}".format(n1, n2, devide_result))    

if __name__ == "__main__":
    main()
