# The main driver code for a forward mode object
# Uses Dual Numbers and elementary_functions.py to continue computation

# Super basic structure, just me spitballing so I don't forget - Hope

class ForwardAD():


    def __init__(self, expression : str, variables):
        # Expression is a string
        # Vars is either a np.array
        self.expression = expression
        self.variables = variables

        self.eval(self)

    def __repr__(self):
        raise NotImplementedError('Devs haven\'t gotten here yet')

    def __str__(self):
        raise NotImplementedError('Devs haven\'t gotten here yet')

    def eval(self):
        # We can convert their variables into into dual numbers with a for loop
        # We can then use this to evaluate the string directly without a parser : 
        #   co = compile(self.expression, '<string>', mode = 'eval')

        # Assuming we then rename all the variables in the expression with a [self.] prefix 
        # Might take some regex to do that

        # Got this idea from Lecture 20 (Code Objects section)

        raise NotImplementedError('Devs haven\'t gotten here yet')

    def jacobian():
        # Construct and print the jacobian
        raise NotImplementedError('Devs haven\'t gotten here yet')








