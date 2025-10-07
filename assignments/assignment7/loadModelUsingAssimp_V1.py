import assimp_py
import numpy as np
import glm
from pathlib import Path
#from math import sqrt
from pathlib import Path
class SceneBound:
    def __init__(self, minmax):
        self.boundingBox = minmax
        self.center = (minmax[0]+ minmax[1])/2
        dVector = (minmax[1] - minmax[0])
        self.radius = glm.length(dVector)/2
    def __str__(self):
        return f"boundingBox:{self.boundingBox}, Center: {self.center}, Radius: {self.radius}."
min_coords = glm.vec3( np.inf)
max_coords = glm.vec3(-np.inf)
verticesList = []
def update_min_max(node, M):
    global min_coords
    global max_coords
    nodeTransform = glm.transpose(glm.mat4(node.transformation));
    currentTransform = M * nodeTransform
    #print(currentTransform)
    if node.num_meshes > 0:
        for index in node.mesh_indices:
            for vertex in verticesList[index]:
                v = (currentTransform*glm.vec4(glm.vec3(vertex), 1)).xyz
                min_coords = glm.min(min_coords,v)
                max_coords = glm.max(max_coords,v)
            #print("Min max: ",min_coords, max_coords)
    # Recurse through child nodes
    for child in node.children:
        update_min_max(child, currentTransform)

def get_bounding_box(scene):
    update_min_max(scene.root_node, glm.mat4(1))
    return SceneBound([min_coords, max_coords])

def printMeshInfo(mesh):
    print("Mesh Name: ",mesh.name)
    print("material Index: ",mesh.material_index)
    print("number of vertices: ",mesh.num_vertices)
    print("number of faces: ",mesh.num_faces)
    print("number of indices: ",mesh.num_indices) 
    print("first 3 verts:\n" + str(np.round(mesh.vertices[:3], decimals=2)))
    print("minimum: ",np.min(mesh.vertices))
    print("first 3 indices:\n" + str(np.round(mesh.indices[:3], decimals=2)))
    print("number of Texture coordinate arrays: ",len(mesh.texcoords))
    for texcoords in mesh.texcoords:
        print("number of Texture coordinates: ",len(texcoords))
        print("first 3 texture coordinates:\n" + str(np.round(texcoords[:3], decimals=2)))

def printMaterialInfo(material):
    # -- getting color
    diffuse_color = material["COLOR_DIFFUSE"]
    print("diffuse_color:", material["COLOR_DIFFUSE"])
    # -- getting textures 
    if material["TEXTURES"]:
        print("diffuse_tex: ", material["TEXTURES"][assimp_py.TextureType_DIFFUSE])
    
def printNodeInfo(node, level): # Recursive call for Depth First Traversal
    print("Hierarchy Level: ",level)
    print("Node name: ", node.name)
    print("Transformation: ",np.round(node.transformation,decimals=2))
    print("Number of meshes: ",node.num_meshes)
    print("Mesh indices: ", node.mesh_indices)
    print("Number of child nodes: ", len(node.children))
    print("-----------------")
    for node in node.children:
        printNodeInfo(node, level+1)
        
def printSceneInfo(scene):
    print("SCENE:")
    print("number of meshes:" , scene.num_meshes)
    print("number of materials:" , scene.num_materials)
    for mesh in scene.meshes:
        printMeshInfo(mesh)
    for material in scene.materials:
        printMaterialInfo(material)
    printNodeInfo(scene.root_node, 0)

def getObjectDataList(filename, verbose=False):
    # -- loading the scene
    process_flags = (
        #assimp_py.Process_Triangulate| assimp_py.Process_CalcTangentSpace
        assimp_py.Process_Triangulate
    )
    folderPath = Path(filename).parent
    scene = assimp_py.import_file(filename, process_flags)
    if verbose:
        printSceneInfo(scene)
    geomDataList = []
    faceIndexDataList = []
    texNames = []
    for mesh in scene.meshes:
        texNames.append(folderPath.name+"/"+scene.materials[mesh.material_index]["TEXTURES"][assimp_py.TextureType_DIFFUSE][0])
        nTextures = len(mesh.texcoords)
        if nTextures==0:
            print("Error: No texture coordinaes in the model data.")
            return
        elif nTextures > 1:
            print(nTextures, " different sets of Texture coordinates. Only the first one is loaded.")
        vertices = np.asarray(mesh.vertices,dtype="f4").reshape(-1,3)
        verticesList.append(vertices)
        geomData = np.concatenate((vertices,np.asarray(mesh.texcoords[0], dtype="f4").reshape(-1,2)), axis=1).astype("float32").flatten()
        geomDataList.append(geomData)
        faceIndexDataList.append(np.array(mesh.indices).astype("int32"))
    return geomDataList, faceIndexDataList, get_bounding_box(scene), texNames, scene