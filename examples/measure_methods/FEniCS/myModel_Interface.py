import subprocess as sp
import numpy as np
import os

def bet_docker_interface(parameter_samples):     
    
    # saves parameter samples to a local file in *.npy format
    np.save("parameter_sample_values",parameter_samples)
    
    # checks to make sure docker is installed and running correctly
    try:
        dockercommand = ("docker version")
        sp.check_call(dockercommand.split(" "))
    except:
        print("\n Error Running Docker: \n Check that Docker "+
          "is properly installed and currently running \n")
        raise
    
    #####
    # checks that appropriate docker FEniCS container exists
    # if the container does not exist, creates the container
    
    # gets list of all docker containers
    dockercommand = ("docker ps -a")
    containerlist = sp.check_output(dockercommand.split(" "))
    #print(containerlist)

    # The name of the FEniCS image: should be downloaded from FEniCS 
    # but creation of container will auto download if not downloaded already
    imagename = ("quay.io/fenicsproject/stable")
    
    # container created by this interface
    containername = ("BET_to_FEniCS")
    
    if not(containername in containerlist):
        print("Container named '"+containername+"' not found."
              +" Creating new Docker container...")
        
        # local directory name
        localdirect = os.getcwd()
        
        # docker create command string
        dockercreate = ("docker create -i --name "+containername
                        + " -w /home/fenics/shared" # sets working directory
                 +" -v "+localdirect+":/home/fenics/shared" # share current dir.
                  +" quay.io/fenicsproject/stable") # name of parent image
        
        # use subprocess to run command string and check output
        sp.check_output(dockercreate.split(" "))
        
        print("New container created named: "+containername)
        
    
    ####
    # Runs the FEniCS model using container
    
    # name of the python script which runs the fenics model
    fenics_script = "fenicsCalculator.py"
    
    # starts container
    dockerstart = ("docker start "+containername)
    outstatus = sp.check_output(dockerstart.split(" "))
    print(outstatus+" container has started...")
    
    # execute python script in FEniCS container
    dockerexec = ("docker exec "+containername+" python "+fenics_script)
    outstatus = sp.Popen(dockerexec.split(" "),stdout=sp.PIPE)
    print(outstatus.communicate()[0])
    
    # close docker container
    dockerclose = ("docker stop "+containername)
    outstatus = sp.check_output(dockerclose.split(" "))
    print(outstatus+" container has closed.")
    
    # Load and save the quantity of interest
    QoI_samples = np.load("QoI_outsample_values.npy")
    
    return QoI_samples 

    
    