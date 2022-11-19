# Interface that the user instantiates to run Auto-Diff
# User should pass in a list / ndarray of funcs (f1, f2, ...) and varLists (x1, x2, ...) but if they pass in single things we can check for that

# They call AutoDiff, passing a pre-defined function(s) and variables as INTs (coordinate to be evaluated at)
# Ex. AutoDiff(f, (2, 6, 5)) where f = x1 + x2 + x3 and we want the derivative at point x1 = 2, x2 = 6, x3 = 5
# They pass variables IN THE ORDER that the function they defined takes them

# There's probably a bunch of other stuff I'm forgetting but please delete, add, or move anything
# Just wanted to get a shell started so everyone can brainstorm the best way to go about this

class AutoDiff:

    def __init__(self, funcs, varList):
        self.funcs = funcs
        self.varList = varList

        self._compute()

    def __repr__(self):
        return f'AutoDiff(functions: {self.funcs}, variables: {self.varList})'
        raise NotImplementedError

    def __str__(self):
        # Are we doing the computational graph here?
        raise NotImplementedError

    def _compute():
        # Driver for AD, mainly just constructing the dual numbers based on the passed vals and then calling the function with those DualNumbers
        # Then make the jacobian and store it in self.jacobian

        # Should we return the jacobian outright? And then also have a getter for when they need the object later
        raise NotImplementedError

    @property
    def get_jacobian(self):
        return self.jacobian

    @property
    def get_varList(self):
        return self.varList

    @property
    def get_funcs(self):
        return self.funcs

    # Do we need setters and deleters? I'm not sure the user would / should use them on an object like this
