import glm
from objloader import Obj
import numpy as np
from math import sqrt

class SceneBound:
    def __init__(self, coords):
        self.boundingBox = [
            np.min(coords,0),
            np.max(coords,0)
        ]
        self.center = (self.boundingBox[0] + self.boundingBox[1])/2
        dVector = (self.boundingBox[1] - self.boundingBox[0])
        self.radius = sqrt(dVector[0]*dVector[0] + dVector[1]*dVector[1]+dVector[2]*dVector[2])/2
    def __str__(self):
        return f"boundingBox:{self.boundingBox}, Center: {self.center}, Radius: {self.radius}."

def get_triangle_normal(vertexList):
    e1 = glm.vec3(vertexList[1]) - glm.vec3(vertexList[0])
    e2 = glm.vec3(vertexList[2]) - glm.vec3(vertexList[0])
    return list(glm.normalize(glm.cross(e1, e2)))

def compute_triangle_normal_coords(position_coord):
    i = 0
    normals = []
    while i < len(position_coord):
        normal = get_triangle_normal(position_coord[i:i+3])
        normals.append(normal)
        normals.append(normal)
        normals.append(normal)
        i += 3
    return np.array(normals)
  
def getObjectData(filePath, normal=False, texture = False):
    geometry = Obj.open(filePath)
    print("Vertex position Count: ",len(geometry.vert),geometry.vert[0])
    if (geometry.norm):
        print("Vertex normal Count: ",len(geometry.norm),geometry.norm[0])
    if (geometry.text):
        print("Vertex texture coordinate Count: ",len(geometry.text),geometry.text[0])
    print("Face count: ", len(geometry.face)/3)
    position_coord = np.array([geometry.vert[f[0]-1] for f in geometry.face])
    if normal==True:
        if geometry.norm:
            normal_coord = np.array([geometry.norm[f[2]-1] for f in geometry.face])
            print("Normal exists")
        else:
            normal_coord = compute_triangle_normal_coords(position_coord)
            print("Normal computed.")
    if texture==True:
        if geometry.text:
            texture_coord = np.array([[geometry.text[f[1]-1][0],geometry.text[f[1]-1][1]] for f in geometry.face])
            print ("texture exists")
        else:
            texture_coord = np.array([[0.5,0.5] for f in geometry.face])
            print("No texture")
    if (normal==False and texture == False):
        vertex_data = position_coord.astype("float32").flatten()
    elif texture == False:
        vertex_data = np.concatenate((position_coord,normal_coord), axis=1).astype("float32").flatten()
    elif normal == False:
        vertex_data = np.concatenate((position_coord,texture_coord), axis=1).astype("float32").flatten()
    else:
        print("All parameters")
        vertex_data = np.concatenate((position_coord,normal_coord,texture_coord), axis=1).astype("float32").flatten()
    return [vertex_data, SceneBound(position_coord)]