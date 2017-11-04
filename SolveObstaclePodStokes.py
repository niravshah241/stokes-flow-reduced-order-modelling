# Copyright (C) 2015-2016 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#
## @file solve_obstacle.py
#  @brief Example 3: geometrical parametrization
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS import *
from dolfin import *
from mshr import *
import time
import os
import shutil # for rm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import itertools

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 3: GEOMETRICAL PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Obstacle(ShapeParametrization(EllipticCoercivePODBase)):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, W, mesh, subd, bound, bc_list):
        # Declare the shape parametrization map
        shape_parametrization_expression = [
            ("(mu[0]-0.5)/0.3*x[1]+x[0]", "mu[1]/0.3*x[1]"), # subdomain 1
            ("(2-2*mu[0])*x[0]+2*mu[0]-1","x[1]+(0.6-2*mu[1])*x[0]+2*mu[1]-0.6"), # subdomain 2
            ("(0.5-mu[0])/0.7*x[1]+x[0]+(mu[0]-0.5)/0.7","(1-mu[1])/0.7*x[1]+(mu[1]-0.3)/0.7"), # subdomain 3
            ("mu[0]/0.5*x[0]","x[1]+(mu[1]-0.3)/0.5*x[0]"), # subdomain 4
            ("(mu[0]-0.5)/0.3*x[1]+x[0]","mu[1]/0.3*x[1]"), # subdomain 5
        ]
        # Call the standard initialization
        super(Obstacle, self).__init__(mesh, subd, W, None, shape_parametrization_expression)      
        # ... and also store FEniCS data structures for assembly
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
        self.bc_list=bc_list
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Set theta multiplicative terms of the affine expansion of a.
   
    def compute_theta_a(self):
        m1 = self.mu[0]
        m2 = self.mu[1]
        # subdomains 1 
        theta_a0 = (1+((m1-0.5)**2)/(m2**2))*m2/0.3 
        theta_a1 = ((0.3/m2)**2)*m2/0.3 
        theta_a2 = (0.3/m2*(0.5-m1)/m2)*m2/0.3 
        theta_a3 = 1*m2/0.3
        theta_a4 = 0
        theta_a5 = -(m1-0.5)/m2*m2/0.3
        theta_a6 = 1 #(=0.3/m2*m2/0.3)
        # subdomains 2 
        theta_a7 = 1/((2-2*m1)**2)*(2-2*m1)
        theta_a8 = (((2*m2-0.6)/(2-2*m1))**2+1)*(2-2*m1)
        theta_a9 = ((2*m2-0.6)/(2-2*m1)**2)*(2-2*m1)
        theta_a10 = 1/(2-2*m1)*(2-2*m1)
        theta_a11 = (2*m2-0.6)
        theta_a12 = 0
        theta_a13 = (2-2*m1)
        # subdomains 3 
        theta_a14 = (1+((m1-0.5)/(1-m2))**2)*(1-m2)/0.7
        theta_a15 = ((0.7/(1-m2))**2)*(1-m2)/0.7
        theta_a16 = (m1-0.5)/(1-m2)
        theta_a17 = (1-m2)/0.7
        theta_a18 = 0
        theta_a19 = (m1-0.5)/0.7
        theta_a20 = 1
        # subdomains 4
        theta_a21 = (0.5/m1)
        theta_a22 = (((m2-0.3)/m1)**2+1)*(m1/0.5)
        theta_a23 = (0.3-m2)/m1
        theta_a24 = 1
        theta_a25 = (0.3-m2)/0.5
        theta_a26 = 0
        theta_a27 = m1/0.5
        # subdomains 5
        theta_a28 = (1+((m1-0.5)**2)/(m2**2))*(m2/0.3) 
        theta_a29 = 0.3/m2
        theta_a30 = ((0.5-m1)/m2) 
        theta_a31 = (m2/0.3)
        theta_a32 = 0
        theta_a33 = (0.5-m1)/0.3
        theta_a34 = 1
        
        return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8, theta_a9, theta_a10, theta_a11, theta_a12, theta_a13, theta_a14, theta_a15, theta_a16, theta_a17, theta_a18, theta_a19, theta_a20, theta_a21, theta_a22, theta_a23, theta_a24, theta_a25, theta_a26, theta_a27, theta_a28, theta_a29, theta_a30, theta_a31, theta_a32, theta_a33, theta_a34)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        m1 = self.mu[0]
        m2 = self.mu[1]
        theta_f0 = m2/0.3 # boundary 1
        theta_f1 = (2-2*m1) # boundary 2
        theta_f2 = (1-m2)/0.7 # boundary 3
        theta_f3 = m1/0.5 # boundary 4
        theta_f4 = m2/0.3 # boundary 5
        # boundaries 1,2,3,4,5
        return (theta_f0, theta_f1, theta_f2, theta_f3, theta_f4)
    ## Set matrices resulting from the truth discretization of a.
    
    def assemble_truth_a(self):
        U = self.u
        V = self.v
        (u,p)=split(U)
        (v,q)=split(V)
        dx = self.dx
        ds = self.ds
        #mu= self.mu[2]
        mu= 0.8
        # subdomains 1  
        a0 = (mu*inner(u.dx(0),v.dx(0)))*dx(1)
        a1 = (mu*inner(u.dx(1),v.dx(1)))*dx(1)
        a2 = (mu*(inner(u.dx(0),v.dx(1))+inner(u.dx(1),v.dx(0))))*dx(1)  
        a3 = (-p*v.dx(0)[0]+q*u.dx(0)[0])*dx(1)#coressponds to delP/delX
        a4 = (-p*v.dx(1)[0]+q*u.dx(1)[0])*dx(1)#coressponds to delP/delY
        a5 = (-p*v.dx(0)[1]+q*u.dx(0)[1])*dx(1)
        a6 = (-p*v.dx(1)[1]+q*u.dx(1)[1])*dx(1)
        # subdomains 2 
        a7 = (mu*inner(u.dx(0),v.dx(0)))*dx(2)
        a8 = (mu*inner(u.dx(1),v.dx(1)))*dx(2)
        a9 = (mu*(inner(u.dx(0),v.dx(1))+inner(u.dx(1),v.dx(0))))*dx(2)  
        a10 = (-p*v.dx(0)[0]+q*u.dx(0)[0])*dx(2)#coressponds to delP/delX
        a11 = (-p*v.dx(1)[0]+q*u.dx(1)[0])*dx(2)#coressponds to delP/delY
        a12 = (-p*v.dx(0)[1]+q*u.dx(0)[1])*dx(2)
        a13 = (-p*v.dx(1)[1]+q*u.dx(1)[1])*dx(2)
        # subdomains 3 
        a14 = (mu*inner(u.dx(0),v.dx(0)))*dx(3)
        a15 = (mu*inner(u.dx(1),v.dx(1)))*dx(3)
        a16 = (mu*(inner(u.dx(0),v.dx(1))+inner(u.dx(1),v.dx(0))))*dx(3)  
        a17 = (-p*v.dx(0)[0]+q*u.dx(0)[0])*dx(3)#coressponds to delP/delX
        a18 = (-p*v.dx(1)[0]+q*u.dx(1)[0])*dx(3)#coressponds to delP/delY
        a19 = (-p*v.dx(0)[1]+q*u.dx(0)[1])*dx(3)
        a20 = (-p*v.dx(1)[1]+q*u.dx(1)[1])*dx(3)
        # subdomains 4 
        a21 = (mu*inner(u.dx(0),v.dx(0)))*dx(4)
        a22 = (mu*inner(u.dx(1),v.dx(1)))*dx(4)
        a23 = (mu*(inner(u.dx(0),v.dx(1))+inner(u.dx(1),v.dx(0))))*dx(4)  
        a24 = (-p*v.dx(0)[0]+q*u.dx(0)[0])*dx(4)#coressponds to delP/delX
        a25 = (-p*v.dx(1)[0]+q*u.dx(1)[0])*dx(4)#coressponds to delP/delY
        a26 = (-p*v.dx(0)[1]+q*u.dx(0)[1])*dx(4)
        a27 = (-p*v.dx(1)[1]+q*u.dx(1)[1])*dx(4)
        # subdomain 5 
        a28 = (mu*inner(u.dx(0),v.dx(0)))*dx(5)
        a29 = (mu*inner(u.dx(1),v.dx(1)))*dx(5)
        a30 = (mu*(inner(u.dx(0),v.dx(1))+inner(u.dx(1),v.dx(0))))*dx(5)  
        a31 = (-p*v.dx(0)[0]+q*u.dx(0)[0])*dx(5)#coressponds to delP/delX
        a32 = (-p*v.dx(1)[0]+q*u.dx(1)[0])*dx(5)#coressponds to delP/delY
        a33 = (-p*v.dx(0)[1]+q*u.dx(0)[1])*dx(5)
        a34 = (-p*v.dx(1)[1]+q*u.dx(1)[1])*dx(5)
        #Boundaries
        
        # Assemble and return
        A0 = assemble(a0, keep_diagonal=True)
        A1 = assemble(a1, keep_diagonal=True)
        A2 = assemble(a2, keep_diagonal=True)
        A3 = assemble(a3, keep_diagonal=True)
        A4 = assemble(a4, keep_diagonal=True)
        A5 = assemble(a5, keep_diagonal=True)
        A6 = assemble(a6, keep_diagonal=True)
        A7 = assemble(a7, keep_diagonal=True)
        A8 = assemble(a8, keep_diagonal=True)
        A9 = assemble(a9, keep_diagonal=True)
        A10 = assemble(a10, keep_diagonal=True)
        A11 = assemble(a11, keep_diagonal=True)
        A12 = assemble(a12, keep_diagonal=True)
        A13 = assemble(a13, keep_diagonal=True)
        A14 = assemble(a14, keep_diagonal=True)
        A15 = assemble(a15, keep_diagonal=True)
        A16 = assemble(a16, keep_diagonal=True)
        A17 = assemble(a17, keep_diagonal=True)
        A18 = assemble(a18, keep_diagonal=True)
        A19 = assemble(a19, keep_diagonal=True)
        A20 = assemble(a20, keep_diagonal=True)
        A21 = assemble(a21, keep_diagonal=True)
        A22 = assemble(a22, keep_diagonal=True)
        A23 = assemble(a23, keep_diagonal=True)
        A24 = assemble(a24, keep_diagonal=True)
        A25 = assemble(a25, keep_diagonal=True)
        A26 = assemble(a26, keep_diagonal=True)
        A27 = assemble(a27, keep_diagonal=True)
        A28 = assemble(a28, keep_diagonal=True)
        A29 = assemble(a29, keep_diagonal=True)
        A30 = assemble(a30, keep_diagonal=True)
        A31 = assemble(a31, keep_diagonal=True)
        A32 = assemble(a32, keep_diagonal=True)
        A33 = assemble(a33, keep_diagonal=True)
        A34 = assemble(a34, keep_diagonal=True)
        return (A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32, A33, A34)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        #mu= self.mu[2]
        mu=0.8
        U=self.u
        (u,p) = split(U)
        V = self.v
        (v,q)=split(V)
        dx = self.dx
        ds = self.ds
        f0 = inner(Constant((0,0)),v)*ds(1)
        f1 = inner(Constant((0,0)),v)*ds(2)
        f2 = inner(Constant((0,0)),v)*ds(3)
        f3 = inner(Constant((0,0)),v)*ds(4)
        f4 = inner(Constant((0,0)),v)*ds(5)
        # boundaries 1,2,3,4,5
        #a12 = inner(u,v)*ds(5) + inner(u,v)*ds(6) + inner(u,v)*ds(7) + inner(u,v)*ds(8)        
        # Assemble and return
        F0 = assemble(f0, keep_diagonal=True)
        F1 = assemble(f1, keep_diagonal=True)
        F2 = assemble(f2, keep_diagonal=True)
        F3 = assemble(f3, keep_diagonal=True)
        F4 = assemble(f4, keep_diagonal=True)
        return (F0,F1,F2,F3,F4)
   
    def compute_error(self, N=None, skip_truth_solve=False):
        if not skip_truth_solve:
            self.truth_solve()
        self.online_solve(N, False)
        self.error.vector()[:] = (self.snapshot.vector()[:] - self.reduced.vector()[:])# error as a function
        self.theta_a = self.compute_theta_a() # not really necessary, for symmetry with the parabolic case
        assembled_truth_A_sym = self.affine_assemble_truth_symmetric_part_matrix(self.truth_A, self.theta_a)
        error_energy_norm_squared = self.compute_scalar(self.error, self.error, assembled_truth_A_sym) # norm of the error
        error_energy_norm = np.sqrt(error_energy_norm_squared)
        relative_error_energy_norm_squared = self.compute_scalar(self.snapshot,self.snapshot,assembled_truth_A_sym)
        relative_error_energy_norm = np.sqrt(relative_error_energy_norm_squared)
        relative_error_energy_norm = error_energy_norm/relative_error_energy_norm
        (v,q) = split(self.error)
        v=project(v,VectorFunctionSpace(mesh, "Lagrange", 2, dim=2))
        q=project(q,FunctionSpace(mesh, "Lagrange", 1))
        (u,p) = split(self.snapshot)
        u=project(u,VectorFunctionSpace(mesh, "Lagrange", 2, dim=2))
        p=project(p,FunctionSpace(mesh, "Lagrange", 1))
        v = norm(v,'H1')
        q = norm(q,'L2')
        u = norm(u,'H1')
        p = norm(p,'L2')
        #error_L2_norm = norm(self.error,'L2')        
        #relative_error_L2_norm = error_L2_norm/norm(self.snapshot,'L2')
        relative_error_H1L2_norm = (v+q)/(u+p)
        return (relative_error_energy_norm, relative_error_H1L2_norm)
                
    def error_analysis(self, N=None):
        self.load_reduced_matrices()
        if N is None:
            N = self.N
            
        self.truth_A = self.assemble_truth_a()
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = self.assemble_truth_f()
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
            
        print ("==============================================================")
        print ("=             Error analysis begins                          =")
        print ("==============================================================")
        print ("")
        
        relative_error_energy_norm = np.zeros((N, len(self.xi_test)))
        relative_error_L2_norm = np.zeros((N,len(self.xi_test)))
        
        for run in range(len(self.xi_test)):
            print ("############################## run = ", run, " ######################################")
            
            self.setmu(self.xi_test[run])
            
            # Perform the truth solve only once
            self.truth_solve()
            
            for n in range(N): # n = 0, 1, ... N - 1
                relative_error_energy_norm[n, run], relative_error_L2_norm[n,run] = self.compute_error(n + 1, True)
        
        mean_mu_relative_error_L2_norm=np.zeros((len(self.xi_test),1))
        mean_mu_relative_error_energy_norm=np.zeros((len(self.xi_test),1))
        print ("                      mu                  \t        mean_mu_relative_error_L2_norm     ")
        
        for r in range(len(self.xi_test)):
            mean_mu_relative_error_L2_norm[r,0] = np.exp(np.mean(np.log((relative_error_L2_norm[self.Nmax-1,r]))))
            print (str(self.xi_test[r]) + " \t " + str(mean_mu_relative_error_L2_norm[r,0]))
        
        print ("                      mu                  \t        mean_mu_relative_error_energy_norm     ")
        
        for r in range(len(self.xi_test)):
            mean_mu_relative_error_energy_norm[r,0] = np.exp(np.mean(np.log((relative_error_energy_norm[self.Nmax-1,r]))))
            print (str(self.xi_test[r]) + " \t " + str(mean_mu_relative_error_energy_norm[r,0]))
            
        # Print some statistics
        mean_N_relative_error_L2_norm=np.zeros((N,1))
        mean_N_relative_error_energy_norm=np.zeros((N,1))
        
        print ("")
        print ("                       N                 \t         mean_N_relative_error_L2_norm  ")
        for n in range(N): # n = 0, 1, ... N - 1
            mean_N_relative_error_L2_norm[n,0] = np.exp(np.mean(np.log((relative_error_L2_norm[n, :]))))
            print (str(n+1)             + " \t " + str(mean_N_relative_error_L2_norm[n,0]))
        
        print ("")
        print ("                       N                 \t         mean_N_relative_error_energy_norm  ")
        for n in range(N): # n = 0, 1, ... N - 1
            mean_N_relative_error_energy_norm[n,0] = np.exp(np.mean(np.log((relative_error_energy_norm[n, :]))))
            print (str(n+1)             + " \t " + str(mean_N_relative_error_energy_norm[n,0]))
            
        self.mean_N_relative_error_energy_norm = mean_N_relative_error_energy_norm
        self.mean_N_relative_error_L2_norm = mean_N_relative_error_L2_norm
        self.mean_mu_relative_error_L2_norm = mean_mu_relative_error_L2_norm
        self.mean_mu_relative_error_energy_norm = mean_mu_relative_error_energy_norm
        
        fig, ax1 = plt.subplots()
        ax1.plot(np.asarray(1+np.asarray(range(N))),np.log10(mean_N_relative_error_L2_norm))
        ax1.set_xlabel('Number of basis function in reduced basis space')
        ax1.set_ylabel('Error in H1L2 norm on logarithmic scale with base 10')
        fig, ax1 = plt.subplots()
        ax1.plot(np.asarray(1+np.asarray(range(N))),np.log10(mean_N_relative_error_energy_norm))
        ax1.set_xlabel('Number of basis function in reduced basis space')
        ax1.set_ylabel('Error in energy norm on logarithmic scale with bse 10')
        
        print ("")
        print ("==============================================================")
        print ("=             Error analysis ends                            =")
        print ("==============================================================")
        print ("")
        
    def generate_train_or_test_set(self, n, sampling):
        if sampling == "uniform":
            ss = "[("
            for i in range(len(self.mu_range)):
                ss += "np.random.uniform(self.mu_range[" + str(i) + "][0], self.mu_range[" + str(i) + "][1])"
                if i < len(self.mu_range)-1:
                    ss += ", "
                else:
                    ss += ") for _ in range(" + str(n) +")]"
            xi = eval(ss)
            return xi
        elif sampling == "linspace":
            n_P_root = int(np.ceil(n**(1./len(self.mu_range))))
            ss = "itertools.product("
            for i in range(len(self.mu_range)):
                ss += "np.linspace(self.mu_range[" + str(i) + "][0], self.mu_range[" + str(i) + "][1], num = " + str(n_P_root) + ").tolist()"
                if i < len(self.mu_range)-1:
                    ss += ", "
                else:
                    ss += ")"
            itertools_xi = eval(ss)
            xi = []
            for mu in itertools_xi:
                xi += [mu]
            return xi
        elif sampling == "random":
            #ss=np.array([np.zeros((n,len(self.mu_range)))])
            #ss=np.asarray([])
            murange=self.mu_range
            for i in range(len(self.mu_range)):
                a=np.squeeze(np.array([(murange[i][1]-murange[i][0])/2*np.random.randn(n,1)+(murange[i][0]+murange[i][1])/2]))
                if i==0:
                    ss=a.reshape(-1,1)
                else:
                    ss=np.hstack((ss,a.reshape(-1,1)))                
            #print(ss)
            return ss
        else:
            sys.exit("Invalid sampling mode.")

        
#  @}
########################### end - PROBLEM SPECIFIC - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 3: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
#mesh = Mesh("domain.xml")

start_time = time.time()
rectangle = Rectangle(Point(0., 0.), Point(+1., +1.))
polygon = Polygon([Point(+0.7, 0.), Point(+0.5, +0.3), Point(+0.3,0.)])

domain = rectangle-polygon

subdomain = dict()
subdomain[1] = Polygon([Point(+1., 0.), Point(+0.5, +0.3), Point(+0.7, 0.0)])
subdomain[2] = Polygon([Point(+1., 0.), Point(+1., +1.), Point(+0.5, +0.3)])
subdomain[3] = Polygon([Point(+1., +1.), Point(0., +1.), Point(+0.5, +0.3)])
subdomain[4] = Polygon([Point(+0.5, +0.3), Point(0., +1.), Point(0., 0.)])
subdomain[5] = Polygon([Point(+0.3, 0.), Point(+0.5, +0.3), Point(0., 0.)])
for i, s in subdomain.iteritems():
    domain.set_subdomain(i, subdomain[i])
mesh = generate_mesh(domain, 10)

subd = MeshFunction("size_t", mesh, 2, mesh.domains())
File("subdomains.pvd")<<subd

tol=1E-10
boundary_markers = FacetFunction("size_t", mesh)
class Boundary1(SubDomain):
    def inside(self,x,on_boundary):
        return (((x[0]>0.5-tol) and (x[0]<0.7+tol) and (x[1]<0.4+tol)) or (x[0]>0.5-tol and near(x[1],0,tol))) and on_boundary 
         
class Boundary2(SubDomain): 
    def inside(self,x,on_boundary):
        return near(x[0],1,tol) and on_boundary
                
class Boundary3(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],1,tol) and on_boundary
        
class Boundary4(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],0,tol) and on_boundary

class Boundary5(SubDomain):
    def inside(self,x,on_boundary):
        return ((x[0]<0.5 and x[0]>0.3-tol and x[1]<0.4+tol) or (x[0]<0.5 and near(x[1],0,tol))) and on_boundary
        
bcOne=Boundary1()
bcOne.mark(boundary_markers,1)    
bcTwo=Boundary2()
bcTwo.mark(boundary_markers,2)
bcThree=Boundary3()
bcThree.mark(boundary_markers,3)
bcFour=Boundary4()
bcFour.mark(boundary_markers,4)
bcFive=Boundary5()
bcFive.mark(boundary_markers,5)

bound = boundary_markers

File("Boundaries.pvd")<<bound

# subdomain and bound

#subd = MeshFunction("size_t", mesh, "subdomains.xml")
#bound = MeshFunction("size_t", mesh, "boundaries.xml")

# 2. Create Finite Element space (Lagrange P1)

V = VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = V*Q
W = FunctionSpace(mesh, TH)

bc1 = DirichletBC(W.sub(0),Constant((0,0)),Boundary1())
bc3 = DirichletBC(W.sub(0),Constant((0,0)),Boundary3())
bc4 = DirichletBC(W.sub(0), Expression(("x[1]*(1-x[1])","0"),degree=2),Boundary4())
bc41 = DirichletBC(W.sub(1),Constant(0),Boundary4())
bc5 = DirichletBC(W.sub(0),Constant((0,0)),Boundary5())


bc_list=[bc1, bc3, bc41, bc4, bc5]

# 3. Allocate an object of the obstacle class
obstacle = Obstacle(W, mesh, subd, bound, bc_list) 

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc' 

# 5. Set mu range, xi_train and Nmax
#mu_range = [(0.3, 0.7), (0.2, 0.3),(0.2, 0.8)]
mu_range = [(0.3, 0.7), (0.2, 0.4)]
obstacle.setmu_range(mu_range)
obstacle.setxi_train(30, False, "linspace")
obstacle.setNmax(25)

# 6. Perform the offline phase
#first_mu = (0.5, 0.3 , 0.4)
first_mu = (0.5, 0.3)
obstacle.setmu(first_mu)
obstacle.offline()

# 7. Perform an online solve
#online_mu = (0.5, 0.3 , 0.42)
online_mu = (0.5,0.3)
obstacle.setmu(online_mu)
start_time=time.time()
obstacle.online_solve()
time_taken = time.time() - start_time
print('time taken for online solution = %s',time_taken)

# 8. Perform an error analysis
obstacle.setxi_test(50, False, "linspace")
obstacle.error_analysis()

# 9. save solution

(u,p)=split(obstacle.snapshot)
u=project(u,VectorFunctionSpace(mesh, "Lagrange", 2, dim=2))
p=project(p,FunctionSpace(mesh, "Lagrange", 1))
'''
obstacle.move_mesh()

plt.figure().suptitle('Velocity_Truth')
plot(u)
plt.figure().suptitle('Pressure_Truth')
plot(p)

ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

obstacle.reset_reference()

(u,p)=split(obstacle.reduced)
u=project(u,VectorFunctionSpace(mesh, "Lagrange", 2, dim=2))
p=project(p,FunctionSpace(mesh, "Lagrange", 1))

obstacle.move_mesh()

ufile_pvd = File("velocity_reduced.pvd")
ufile_pvd << u
pfile_pvd = File("pressure_reduced.pvd")
pfile_pvd << p

plt.figure().suptitle('Velocity_Reduced')
plot(u)
plt.figure().suptitle('Pressure_Reduced')
plot(p)

obstacle.reset_reference()
'''
a=np.load('xi_test__pod/xi_test.npy')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[:,0],a[:,1],np.log10(obstacle.mean_mu_relative_error_L2_norm[:,0]))
ax.set_xlim(np.amin(a[:,0]), np.amax(a[:,0]))
ax.set_ylim(np.amin(a[:,1]), np.amax(a[:,1]))
ax.set_zlim(np.log10(np.amin(obstacle.mean_mu_relative_error_L2_norm[:,0])),np.log10(np.amax(obstacle.mean_mu_relative_error_L2_norm[:,0])))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('mean_mu_relative_error_H1L2_norm on log scale with base 10')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[:,0],a[:,1],np.log10(obstacle.mean_mu_relative_error_energy_norm[:,0]))
ax.set_xlim(np.amin(a[:,0]), np.amax(a[:,0]))
ax.set_ylim(np.amin(a[:,1]), np.amax(a[:,1]))
ax.set_zlim(np.log10(np.amin(obstacle.mean_mu_relative_error_energy_norm[:,0])),np.log10(np.amax(obstacle.mean_mu_relative_error_energy_norm[:,0])))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('mean_mu_relative_error_energy_norm on log scale with base 10')

b=np.load('post_processing__pod/eigs.npy')
fig, ax1 = plt.subplots()
ax1.plot(np.log10(b))
ax1.set_xlabel('')
ax1.set_ylabel('Eigenvalue on logarithmic scale with base 10')

plt.show()

#print("training set parameters %s"%np.load('xi_train__pod/xi_train.npy'))
#print("test set parameters %s"%np.load('xi_test__pod/xi_test.npy'))