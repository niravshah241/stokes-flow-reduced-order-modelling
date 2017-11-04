from dolfin import *
from mshr import *

import matplotlib.pyplot as plt
import time

# Create mesh
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
#subdomain[6] = Polygon([Point(+2., -2.), Point(+2., +2.), Point(+1., +1.)])
#subdomain[7] = Polygon([Point(-2., +2.), Point(-1., +1.), Point(+1., +1.)])
#subdomain[8] = Polygon([Point(-2., +2.), Point(+1., +1.), Point(+2., +2.)])

for i, s in subdomain.iteritems():
    domain.set_subdomain(i, subdomain[i])
mesh = generate_mesh(domain, 10)
#plot(mesh)
#plt.figure()
#plt.show()
#interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
plot(subdomains)
plt.show()
#interactive()

# Create boundaries
tol=1E-14
boundary_markers = FacetFunction("size_t", mesh)
class Boundary1(SubDomain):
    def inside(self,x,on_boundary):
        return (((x[0]>0.5-tol) and (x[0]<0.7+tol) and (x[1]<0.3+tol)) or (x[0]>0.7-tol and near(x[1],0,tol))) and on_boundary
            
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
        return ((x[0]<0.5+tol and x[0]>0.3-tol and x[1]<0.3+tol) or (x[0]<0.3-tol and near(x[1],0,tol))) and on_boundary

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

#plot(boundary_markers)
#plt.show()
#interactive()

# Save
File("domain.xml") << mesh
File("domain.pvd")<<mesh
File("subdomains.xml") << subdomains
File("subdomains.pvd") << subdomains
File("boundaries.xml") << boundary_markers
File("boundaries.pvd") << boundary_markers

print("\n --- Time elapsed for mesh generation = %s --- \n" %(time.time()-start_time))
plot(boundary_markers)
plt.show()