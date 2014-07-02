# -*- coding: utf-8 -*-
def vorbonvon(pts_per_edge, lambda_domain):


    # add center points (cells that make up R)
    
    
    # add first surrounding layer (cells that surround R to make up
    # lambda_domain)

    # add second surrounding layer (cells that bound lambda_domain)


    # Calcuate the bounding region for the parameters
    lam_bound = np.copy(samples)
    lam_width = lam_domain[:,1]-lam_domain[:,0]
    # Add fake samples outside of lam_domain to close Voronoi tesselations.
    sides = np.zeros((2,pts_per_edge))
    for i in range(lam_domain.shape[0]):
        sides[i,:] = np.linspace(lam_domain[i,0], lam_domain[i,1], pts_per_edge)
    # add midpoints
    for i in range(lam_domain.shape[0]):
        new_pt = sides
        new_pt[i,:] = np.repeat(lam_domain[i,0]-lam_width[i]/pts_per_edge,
                pts_per_edge,0).transpose() 
        lam_bound = np.vstack((lam_bound,new_pt))
        new_pt = sides
        new_pt[i,:] = np.repeat(lam_domain[i,1]-lam_width[i]/pts_per_edge,
                pts_per_edge,0).transpose() 
        lam_bound = np.vstack((lam_bound,new_pt))
          
    # add corners
    corners = np.zeros((2**lam_domain.shape[0], lam_domain.shape[0]))
    for i in range(lam_domain.shape[0]):
        corners[i,:] = lam_domain[i,np.repeat(np.hstack((np.ones((1,2**(i-1))),
            2*np.ones((1,2**(i-1))))),
            2**(lam_domain.shape[0]-i),0).transpose()] 
        corners[i,:] +=lam_width[i]*np.repeat(np.hstack((np.ones((1,2**(i-1))),
            -np.ones((1,2**(i-1))))),
            2**(lam_domain.shape[0]-i)/pts_per_edge,0).transpose()

    lam_bound = np.vstack((lam_bound, corners))
      
    # Calculate the Voronoi diagram for samples. Calculate the volumes of 
    # the convex hulls of the corresponding Voronoi regions.
    lam_vol = np.zeros((samples.shape[-1],))
    for i in range((samples.shape[0])):
        vornoi = spatial.Voronoi(lam_bound)
        lam_vol[i] = float(pyhull.qconvex('Qt FA', vornoi.vertices).split()[-1])
