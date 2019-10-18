import argparse

def main():
    parser = argparse.ArgumentParser()
	
    #adding arguments (positonal argument)
    parser.add_argument("number1", help = "add number1")
    

    #adding arguments (optional argument)
	
    #argument type: 1
    #remember by default datatype of passed argument is string, unless "type" mentioned while adding argument
    parser.add_argument("-n2","--number2", help = "add number2. it is optional", type=int)
    
	
    #argument type: 2
    #action while addargument
    #The option is now more of a flag than something that requires a value.
    #We even changed the name of the option to match that idea.
    #Note that we now specify a new keyword, action, and give it the value "store_true".
    #This means that, if the option is specified, assign the value True to args.
    #verbose. Not specifying it implies False.
    #complains when you specify a value, in true spirit of what flags actually are.
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	
    #argument type: 3
    parser.add_argument("-x", action="count",  default=0, help="increase output verbosity")
    
    #argument type: 4
    parser.add_argument("-i", type=str, choices = ["A", "B", "C"], help = "string input from the given choice")
            
    #here all the arguments get saved in the object "args"
    #we could access those the way you access atribute of an object
    args = parser.parse_args()
        
    print(args.number1)
    print(args.number2)
    if args.verbose:
        print("verbosity turned on")
    print(args.i)
    print(args.x)
	


if __name__ == "__main__":
    main()

print("all done")
#python3 argparse_test.py -h
#python3 argparse_test.py -help
#python3 argparse_test.py 4 --number2 5
#python3 argparse_test.py 4
#python3 argparse_test.py 4 -v
#python3 argparse_test.py 4 -i A
#python3 argparse_test.py 4 -xx
#will get 2 because xx is two time
