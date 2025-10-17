import assimp_py
import numpy as np
import glm
from pathlib import Path
import pygame

class SceneBound:
    def __init__(self, minmax):
        self.boundingBox = minmax
        self.center = (minmax[0]+ minmax[1])/2
        dVector = (minmax[1] - minmax[0])
        self.radius = glm.length(dVector)/2
    def __str__(self):
        return f"boundingBox:{self.boundingBox}, Center: {self.center}, Radius: {self.radius}."

class create3DAssimpObject:

    def _update_min_max(self,node, M):
        nodeTransform = glm.transpose(glm.mat4(node.transformation));
        currentTransform = M * nodeTransform
        #print(currentTransform)
        if node.num_meshes > 0:
            for index in node.mesh_indices:
                for vertex in self._verticesList[index]:
                    v = (currentTransform*glm.vec4(glm.vec3(vertex), 1)).xyz
                    self._min_coords = glm.min(self._min_coords,v)
                    self._max_coords = glm.max(self._max_coords,v)
                #print("Min max: ",min_coords, max_coords)
        # Recurse through child nodes
        for child in node.children:
            self._update_min_max(child, currentTransform)

    def _get_bounding_box(self):
        self._min_coords = glm.vec3( np.inf)
        self._max_coords = glm.vec3(-np.inf)
        self._update_min_max(self.scene.root_node, glm.mat4(1))
        return SceneBound([self._min_coords, self._max_coords])

    def _printMeshInfo(self,mesh):
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
            print("first 2 texture coordinates:\n" + str(np.round(texcoords[:2], decimals=2)))

    def _printMaterialInfo(self,material):
        # -- getting color
        # -- getting textures 
        if "NAME" in material.keys():
            print("Material Name: ", material["NAME"])
        else:
            print("No material Name")
        if "TEXTURE_BASE" in material.keys():
            print("Base Texture: ", material['TEXTURE_BASE'])
        else:
            print("No Base Texture")
        if material["TEXTURES"]:
            print("diffuse_tex: ", material["TEXTURES"][assimp_py.TextureType_DIFFUSE])
        print("Diffuse_color: ", material["COLOR_DIFFUSE"])
        print("Shininess: ",material["SHININESS"])
        if material["TEXTURES"]:
            print("Diffuse Texture: ",material["TEXTURES"][assimp_py.TextureType_DIFFUSE])
            if assimp_py.TextureType_SPECULAR in material["TEXTURES"].keys():
                print("Specular Texture: ",material["TEXTURES"][assimp_py.TextureType_SPECULAR])
            else:
                print("No specular texture")
        
         #print(material)
           
    def _printNodeInfo(self,node, level): # Recursive call for Depth First Traversal
        print("Hierarchy Level: ",level)
        print("Node name: ", node.name)
        print("Transformation: ",np.round(node.transformation,decimals=2))
        print("Number of meshes: ",node.num_meshes)
        print("Mesh indices: ", node.mesh_indices)
        print("Number of child nodes: ", len(node.children))
        print("-----------------")
        for node in node.children:
            self._printNodeInfo(node, level+1)
            
    def _printSceneInfo(self):
        print("SCENE:")
        print("number of meshes:" , self.scene.num_meshes)
        for mesh in self.scene.meshes:
            self._printMeshInfo(mesh)
        print("number of materials:" , self.scene.num_materials)
        for i,material in enumerate(self.scene.materials):
            print("Material Index: ", i)
            self._printMaterialInfo(material)
        self._printNodeInfo(self.scene.root_node, 0)
        
    def __init__(self, filename, verbose=False, process_flags=(assimp_py.Process_Triangulate), normalFlag = False, textureFlag = False, tangentFlag = False ):
        #[attr for attr in dir(assimp_py) if not attr.startswith('__') and not callable(getattr(assimp_py, attr))]
        self._min_coords = glm.vec3( np.inf)
        self._max_coords = glm.vec3(-np.inf)
        #process_flags = ( assimp_py.Process_Triangulate| assimp_py.Process_CalcTangentSpace )
        filepath = Path(filename)
        folderPath = filepath.parent
        
        self.scene = assimp_py.import_file(filename, process_flags)

        if normalFlag==True:
            if self.scene.meshes[0].normals:
                print("Accumulates Normals in the geomDataList")
            else:
                print("Error: Model does not have Normals. You may make a call with process_flag assimp_py.Process_GenNormals, or assimp_py.Process_GenSmoothNormals.")
                return
            
        if textureFlag==True:
            if self.scene.meshes[0].texcoords and (len(self.scene.meshes[0].texcoords) > 0):
                print("Accumulates Textures in the geomDataList")
            else:
                print("Error: Model does not have Texture Coordinates. You may try making a call with process_flag assimp_py.Process_GenUVCoords.")
                return
            
        if tangentFlag == True:
            if self.scene.meshes[0].tangents:
                print("Accumulates Tangents in the geomDataList")
            else:
                print("Error: Model does not have Tangents. You may make a call with process_flag assimp_py.Process_CalcTangentSpace.")
                return

        if verbose:
            print("File: ",filepath.name)
            print("Folder Path: ", folderPath.name)
            self._printSceneInfo()
        
        self._geomDataList = []
        self._faceIndexDataList = []
        self._verticesList = []
        self.texNames = []
        
        for index, mesh in enumerate(self.scene.meshes):
            #print("Mesh ", index, "material Index: ",mesh.material_index)
            vertices = np.asarray(mesh.vertices,dtype="f4").reshape(-1,3)
            #print("Num Vertices: ",len(vertices))
            self._verticesList.append(vertices)
            normals = None
            texCoords = None
            tangents = None
            if normalFlag:
                #print("Normal size:" ,len(mesh.normals))
                vertices = np.concatenate((vertices, np.asarray(mesh.normals, dtype="f4").reshape(-1,3)), axis=1)
            if textureFlag:
                nTextures = len(mesh.texcoords)
                #print(" number of texture coordinates array: ", nTextures)
                if nTextures==0:
                    self.texNames.append(None)
                    print("Error: No texture coordinates in the model data.")
                    return
                elif nTextures > 1:
                    print(nTextures, " different sets of Texture coordinates. Only the first one is loaded.")
                textureInfo = self.scene.materials[mesh.material_index].get("TEXTURE_BASE")
                #print(textureInfo[assimp_py.TextureType_DIFFUSE])
                if textureInfo:
                    texName = ((folderPath.name+"/") if folderPath.name!="" else "") + textureInfo#textureInfo[assimp_py.TextureType_DIFFUSE][0]
                    self.texNames.append(texName)
                else:
                    self.texNames.append(None)            
                vertices = np.concatenate((vertices,np.asarray(mesh.texcoords[0], dtype="f4").reshape(-1,2)), axis=1)
            if tangentFlag:
                #print("Tangents Size: ",len(mesh.tangents))
                vertices = np.concatenate((vertices,np.asarray(mesh.tangents, dtype="f4").reshape(-1,3)), axis=1)
            geomData = vertices.flatten()
            self._geomDataList.append(geomData)
            self._faceIndexDataList.append(np.array(mesh.indices).astype("int32"))
        self.bound = self._get_bounding_box()

    def getRenderables(self, gl, program, format, variables):
        renderables = []

        for i,geomData in enumerate(self._geomDataList):

            geomBuffer = gl.buffer(geomData)
            indexBuffer = gl.buffer(self._faceIndexDataList[i])

            renderable = gl.vertex_array(program,
                [(geomBuffer, format, *variables)],
                index_buffer=indexBuffer,index_element_size=4
            )
            renderables.append(renderable)
        return renderables
        

    def getSamplers(self,gl):
        # Create a dummy texture
        noTextureSampler = gl.sampler(
            texture = gl.texture(size=(1, 1),data=np.array([255, 0, 0, 255], dtype=np.uint8).tobytes(),components=4), 
            filter=(gl.NEAREST, gl.NEAREST), repeat_x = False, repeat_y = False
        )
        samplers = []

        uniqueTexNames = []
        uniqueSamplers = []
        def getIndexInUniqueList(name):
            index = 0
            try:
                index = uniqueTexNames.index(name)
            except ValueError:
                index = -1
            return index

        for i, texName in enumerate(self.texNames):
            if texName == None:
                samplers.append(noTextureSampler)
            else:
                index = getIndexInUniqueList(texName)
                if index == -1:
                    texture_img = pygame.image.load(texName) 
                    texture_data = pygame.image.tobytes(texture_img,"RGB", True) 
                    texture = gl.texture(texture_img.get_size(), data = texture_data, components=3)
                    texture.build_mipmaps()
                    sampler = gl.sampler(texture=texture, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x = True, repeat_y = True)
                    uniqueSamplers.append(sampler)
                    uniqueTexNames.append(texName)
                    samplers.append(sampler)
                else:
                    samplers.append(uniqueSamplers[index])
        return samplers
        
    def _recursive_render(self, node, M, samplers, renderables, program):
        nodeTransform = glm.transpose(glm.mat4(node.transformation))
        M1 = M * nodeTransform
        if node.num_meshes > 0:
            for index in node.mesh_indices:
                samplers[index].use(0)
                program["map"] = 0
                program["model"].write(M1)
                renderables[index].render() 
        for child in node.children:
            self._recursive_render(child, M1, samplers, renderables, program)
    def render(self, program, renderables, samplers, M = glm.mat4(1)):
        self._recursive_render(self.scene.root_node, M, samplers, renderables, program)