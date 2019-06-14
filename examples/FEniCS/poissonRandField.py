from dolfin import*


def solvePoissonRandomField(rand_field, V, f, bcs):
    """
    Solves the poisson equation with a random field :
    (\grad \dot (rand_field \grad(u)) = -f)
    """
    # create the function space
    u = TrialFunction(V)
    v = TestFunction(V)
    L = f*v*dx
    a = inner(rand_field*nabla_grad(u), nabla_grad(v))*dx
    u = Function(V)
    solve(a == L, u, bcs)
    return u
