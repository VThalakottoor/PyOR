"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain function related to Rotation
"""

def RotateX(self,theta):
    """
    Rotation about X
    """
    theta = theta * np.pi / 180.0
    return np.asarray([[1,0,0],[0, np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def RotateY(self,theta):
    """
    Rotation about Y
    """
    theta = theta * np.pi / 180.0
    return np.asarray([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    
def RotateZ(self,theta):
    """
    Rotation about Z
    """
    theta = theta * np.pi / 180.0
    return np.asarray([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def RotateEuler(self,alpha,beta,gamma):
    """
    Euler Angles
    """
    return self.RotateZ(alpha) @ self.RotateY(beta) @ self.RotateZ(gamma)