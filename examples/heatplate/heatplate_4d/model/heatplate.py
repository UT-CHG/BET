# Copyright (C) 2014-2015  BET Development Team

"""Temperature variations of a thin (2-D) plate with a heat source 'underneath' and perfectly insulated (Neumann boundaries set to 0).  We turn the heat source off after time t_heatoff"""

from dolfin import *
import numpy
from simulation_4kappas_setup import *


def heatplatecenter(amp, px, py, width, degree, T_R, kappavec, rho, cap, nx, ny, mesh, functional_list, radii, angles, corner_radii, corner_angles, dt, t_stop):

    #define the subspace we will solve the problem in
    V = FunctionSpace(mesh, 'Lagrange', 1)

    #turn off the heat halfway through
    t_heatoff = t_stop/2.0

    #time stepping method. forward, backward, CN, etc...
    theta = 0.5
    
    # Define 9 regions on the plate for different kappa values
    class Kappa(Expression):
        def eval(self, value, x):
            """x: spatial point, value[0]: function value."""
            material = 0
            if x[0]>=0 and x[1]>=0:
                material = 0
            elif x[0]<0 and x[1]>=0:
                material = 1
            elif x[0]<0 and x[1]<0:
                material = 2
            elif x[0]>=0 and x[1]<0:
                material = 3

            value[0] = kappavec[material]

    kappa = Kappa()
    

    # Define initial condition(initial temp of plate)
    T_1 = interpolate(Constant(T_R), V)

    # Define variational problem
    T = TrialFunction(V)

    #two f's and L's for heat source on and off
    f_heat = Expression('amp*exp(-((x[1]-py)*(x[1]-py)+(x[0]-px)*(x[0]-px))/width)', amp=amp, px=px, py=py, width=width)
    f_cool = Constant(0)
    v = TestFunction(V)
    a = rho*cap*T*v*dx + theta*dt*kappa*inner(nabla_grad(v), nabla_grad(T))*dx
    L_heat = (rho*cap*T_1*v + dt*f_heat*v - (1-theta)*dt*kappa*inner(nabla_grad(v), nabla_grad(T_1)))*dx
    L_cool = (rho*cap*T_1*v + dt*f_cool*v - (1-theta)*dt*kappa*inner(nabla_grad(v), nabla_grad(T_1)))*dx
    
    A = assemble(a)
    b = None  # variable used for memory savings in assemble calls

    T = Function(V) 
    t = dt
   
    #time stepping
    while t <= t_stop:
        relerrorvec = []
        #print 'time =', t
        if t < t_heatoff:
            b = assemble(L_heat, tensor=b)
        else:
            b = assemble(L_cool, tensor=b)
        solve(A, T.vector(), b)

        t += dt
        T_1.assign(T) 
        '''
        This section uses a crude numerical quadrature to approximate functionals of the solution
        The functionals represent an approximate temperature measurement at some point in space
        #####################################################
        eval T(x,y) using an average of the function over a small disc about (x,y)
        gather temps on concentric cirles (radii and angles passed in)
        '''
        count = 1
        E = interpolate(Constant(1.0), V)
        for i in range(len(radii)):
            for j in range(len(angles)):
                fx = radii[i]*cos(angles[j])
                fy = radii[i]*sin(angles[j])
                r = 0.05
                num_slices = 4
                area_disc = pi*r**2
                area_slice = pi*r**2/num_slices
                angle_weights = (2*pi)/(2*num_slices)
                integral_sum = 0
                for k in range(num_slices):
                    integral_sum = integral_sum + area_slice*T(fx+(r/2)*cos(k*angle_weights), fy+(r/2)*sin(k*angle_weights))

                T_functional_approx = 1.0/(area_disc)*integral_sum
                functional_list.append(T_functional_approx)

                count += 1

        #gather temps near corners
        for i in range(len(corner_radii)):
            for j in range(len(corner_angles)):
                fx = corner_radii[i]*cos(corner_angles[j])
                fy = corner_radii[i]*sin(corner_angles[j])
                r = 0.05
                num_slices = 4
                area_disc = pi*r**2
                area_slice = pi*r**2/num_slices
                angle_weights = (2*pi)/(2*num_slices)
                integral_sum = 0
                for k in range(num_slices):
                    integral_sum = integral_sum + area_slice*T(fx+(r/2)*cos(k*angle_weights), fy+(r/2)*sin(k*angle_weights))

                T_functional_approx = 1/(area_disc)*integral_sum
                functional_list.append(T_functional_approx)

                count += 1
       
    #plot(T)
    #nteractive()










