
import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V2 import create3DAssimpObject

width = 840
height = 480

pygame.init() 
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE) 
pygame.display.set_mode((width, height), flags= pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption(title = "Project Assignment 01: Caitlin Box")
gl = moderngl.get_context() 
gl.info["GL_VERSION"]

#
# Vertex shader(s)
#
vertex_shader_code = '''
#version 460 core
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;


uniform mat4 model, view, perspective;
uniform mat3 normalMatrix;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;
void main() {
    f_uv = uv;
    vec4 P = model*vec4(position, 1);
    f_position = P.xyz;
    gl_Position = perspective*view*P;
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    f_normal = normalize(normalMatrix*normal);
}
'''

#
# Fragment shader(s)
#
fragment_shader_code = '''
#version 460 core

in vec3 f_normal;
in vec3 f_position;
in vec2 f_uv;

uniform sampler2D map;
uniform samplerCube cubeEnv;
uniform vec3 eye_position;
uniform bool metal;

uniform vec3 skyColor;
uniform vec3 groundColor;
uniform vec3 upDir;

out vec4 out_color;

void main() {
    vec3 N = normalize(f_normal);
    vec3 V = normalize(eye_position - f_position);

    if (metal) {
        vec3 R = reflect(-V, N);
        vec3 envColor = texture(cubeEnv, normalize(R)).rgb;

        vec3 specTint = texture(map, f_uv).rgb;
        vec3 color = specTint * envColor;

        out_color = vec4(color, 1.0);
    } else {
        float w = 0.5 * (1.0 + dot(N, normalize(upDir)));
        w = clamp(w, 0.0, 1.0);
        vec3 incident = mix(groundColor, skyColor, w);

        vec3 base = vec3(1.0);
        vec3 color = incident * base;
        out_color = vec4(color, 1.0);
    }
}   
'''
#
# Programs
#

program_model = gl.program(
    vertex_shader= vertex_shader_code,
    fragment_shader= fragment_shader_code
)
format = "3f 3f 2f"
variables = ["position","normal", "uv"]

model_file = "the_utah_teapot/scene.gltf" 
model_obj = create3DAssimpObject(model_file, verbose=False, textureFlag = True, normalFlag = True)

model_renderable = model_obj.getRenderables(gl, program_model, format, variables)
scene = model_obj.scene

def recursive_render(node, M):
    nodeTransform = glm.transpose(glm.mat4(node.transformation));
    currentTransform = M * nodeTransform
    if node.num_meshes > 0:
        for index in node.mesh_indices:
            model_renderable[index]._program["model"].write(currentTransform)         
            model_renderable[index].render()
            
    for node in node.children:
        recursive_render(node, currentTransform)

def render():
    recursive_render(scene.root_node, M=glm.mat4(1))
    
_imageFile = "gold.jpg"
_texture_img = pygame.image.load(_imageFile) 
_texture_data = pygame.image.tobytes(_texture_img,"RGB", True) 
_texture = gl.texture(_texture_img.get_size(), data = _texture_data, components=3)
_texture.build_mipmaps()
gold_sampler = gl.sampler(texture=_texture, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)
bound = model_obj.bound

def cubemap_sampler(base_dir="Footballfield"):
    sides = ["posx","negx","posy","negy","posz","negz"]
    byte_array = bytearray()
    last_size = None
    for f in sides:
        path = f"{base_dir}/{f}.jpg"
        img = pygame.image.load(path)
        if last_size is None:
            last_size = img.get_size()
        else:
            assert img.get_size() == last_size, "All cubemap face images must have the same size"
        data = pygame.image.tobytes(img, "RGB", False)
        byte_array.extend(data)
    tex = gl.texture_cube(last_size, components=3, data=byte_array)
    tex.build_mipmaps()
    return gl.sampler(texture=tex, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)

cube_sampler = cubemap_sampler()

_positions = numpy.array([
    [-1, 1],
    [ 1, 1],
    [ 1,-1],
    [-1,-1]
]).astype("float32")

_geom = _positions.flatten()

_index = numpy.array([
    0, 1, 2,
    2, 3, 0
]).astype("int32")

#
# Vertex shader(s)
#
_vertex_shader_code = '''
#version 460 core
in vec2 position;

uniform mat4 inversepm;
uniform vec3 eye_position;

out vec3 Vdir;

void main() {
    vec4 clipPos = vec4(position, 1.0, 1.0);
    gl_Position = clipPos;

    vec4 P = inversepm * clipPos;
    vec3 world = P.xyz / P.w;
    Vdir = normalize(world - eye_position);
 }
'''
#
# Fragment shader(s)
#
_fragment_shader_code = '''
#version 460 core
in vec3 Vdir;
uniform samplerCube cubeEnv;

out vec4 out_color;

void main() {
    vec3 c = texture(cubeEnv, normalize(Vdir)).rgb;
    out_color = vec4(c, 1.0);
}

'''
football_program = gl.program(
    vertex_shader= _vertex_shader_code,
    fragment_shader= _fragment_shader_code
)

football_renderable = gl.vertex_array(football_program,
                [(gl.buffer(_geom), "2f", "position")],
                index_buffer=gl.buffer(_index),index_element_size=4
            )

displacement_vector = 2*bound.radius*glm.rotate(glm.vec3(0,1,0), glm.radians(85), glm.vec3(1,0,0)) #glm.vec3(0,0,1) 

light_displacement_vector = 2*bound.radius*glm.rotate(glm.vec3(0,1,0), glm.radians(45), glm.vec3(1,0,0)) 
    
target_point = glm.vec3(bound.center)
up_vector = glm.vec3(0,1,0)

### View volume parameters
fov_radian = glm.radians(45) 
aspect = width/height
near = bound.radius
far = 3*bound.radius
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

running = True
clock = pygame.time.Clock()
alpha = 0
lightAngle = 0

pause = True 
metal = False  
skybox = False 

gl.depth_func = '<=' 
gl.enable(gl.DEPTH_TEST)

program_model["skyColor"].write(glm.vec3(0.718, 0.741, 0.753))
program_model["groundColor"].write(glm.vec3(0.322, 0.4, 0.11))
program_model["upDir"].write(glm.vec3(0.0, 1.0, 0.0))

while running:   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif (event.type ==  pygame.KEYDOWN):
            if  event.key == 27:
                running = False
            elif event.key == pygame.K_p:
                pause = not pause
            elif event.key == pygame.K_s:
                skybox = not skybox
            elif event.key == pygame.K_m:
                metal = not metal
            elif event.key == pygame.K_LEFT:
                lightAngle -= 5
            elif event.key == pygame.K_RIGHT:
                lightAngle += 5
        elif (event.type == pygame.WINDOWRESIZED):
            width = event.x
            height = event.y
            aspect = width / height if height > 0 else 1.0
            perspectiveMatrix = glm.perspective(fov_radian, width/height, near, far)
            gl.viewport = (0, 0, width, height)

    # create the aspect ratio correction matrix
    new_displacement_vector = glm.rotate(displacement_vector, glm.radians(alpha), glm.vec3(0,1,0))

    new_light_displacement_vector = glm.rotate(light_displacement_vector, glm.radians(lightAngle), glm.vec3(0,1,0))
    
    eye_point = target_point + new_displacement_vector

    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    gl.clear(0.5,0.5,0.0, depth=1.0)
    gl.depth_func = '<=' 

    if skybox:
        pm = perspectiveMatrix * viewMatrix
        inverse_pm = glm.inverse(pm)

        football_program["inversepm"].write(inverse_pm)
        football_program["eye_position"].write(eye_point)
        cube_sampler.use(1)
        football_program["cubeEnv"].value = 1

        gl.depth_mask = False
        gl.disable(moderngl.DEPTH_TEST)
        football_renderable.render()

        gl.enable(moderngl.DEPTH_TEST)
        gl.depth_mask = True
        gl.depth_func = '<=' 

    # Render Relfector
    program = program_model
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)   
    program["eye_position"].write(eye_point)
    program["metal"].value = 1 if metal else 0

    gold_sampler.use(0)
    cube_sampler.use(1)
    program["map"] = 0
    program_model["cubeEnv"].value = 1
    render()
    
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 10
    if not pause:
        alpha +=  1
        if alpha > 360:
            alpha = 0
pygame.display.quit()