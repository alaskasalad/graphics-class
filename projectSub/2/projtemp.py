import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V3 import create3DAssimpObject
# For stenciling you must install pyOpenGL via pip install pyOpenGL and uncomment the following statement.
from OpenGL.GL import * 

width = 840
height = 480

pygame.init() # Initlizes its different modules. Display module is one of them.
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1) 
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE) 
pygame.display.set_mode((width, height), flags= pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption(title = "Class Practice: Instructor")

gl = moderngl.get_context() # Get previously created context.
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

out vec2 f_uv;
out vec3 f_normal; // Normal vector in World Coordinates
out vec3 f_position; // postion in world coordinates

void main() {
    vec4 P = model*vec4(position, 1);
    f_position = P.xyz;

    f_uv = uv;
    mat3 normalMatrix = mat3(transpose(inverse(model)));// inverse transpose of model transformation
    f_normal = normalize(normalMatrix*normal);

    gl_Position = perspective*view*P;
 }
'''

#
# Fragment shader(s)
#
fragment_shader_code = '''
    #version 430 core
    in vec2 f_uv;
    in vec3 f_normal;
    in vec3 f_position;
    
    uniform sampler2D map;
    uniform vec4 light;
    
    uniform float shininess;
    uniform vec3 eye_position;
    uniform vec3 k_diffuse;

    const vec3 up = vec3(0, 1, 0);
    const vec3 groundColor = vec3(0.3215686274509804,0.4,0.10980392156862745);
    const vec3 skyColor = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);
    
    // Add output variable here
    out vec4 out_color;
    
    vec3 computeColor(){
        vec3 L = normalize(light.xyz);
        if (light.w>0)
            L = normalize(light.xyz-f_position);
        vec3 materialColor = texture(map, f_uv).rgb;
        vec3 N = normalize(f_normal);
        float NdotL = dot(N,L);
        vec3 color = vec3(0.);
        float w = dot(N,up);
        vec3 ambientColor = 0.25*(w*skyColor+(1-w)*groundColor)*materialColor;
        if (NdotL>0.){
            vec3 diffuselyReflectedColor = materialColor * NdotL;
            // Compute specular color
            vec3 V = normalize(eye_position - f_position);
            vec3 H = normalize(L+V);
            vec3 specularlyReflectedColor = vec3(0);
            if (shininess > 0)
                specularlyReflectedColor = vec3(1)*pow(dot(N,H), shininess);
            color = k_diffuse * diffuselyReflectedColor + specularlyReflectedColor;
        }
        color += ambientColor; 
        return color;
    }
    void main() {
        out_color = vec4(computeColor(), 1);
    }
'''
#
# Programs
#

model_program = gl.program(
    vertex_shader= vertex_shader_code,
    fragment_shader= fragment_shader_code
)
format = "3f 3f 2f"
variables = ["position", "normal", "uv"]

### Use Assimp parser and read a 3D model.
modelFile = "mario_obj/scene.gltf" 
modelObj = create3DAssimpObject(modelFile, verbose=False, textureFlag = True, normalFlag = True)

#model_renderables = modelObj.getRenderables(gl, model_program, format, variables)
modelObj.createRenderableAndSampler(model_program)

bound = modelObj.bound

def render_model(view, perspective,light, eye):
    # Render Relfector
    program = model_program
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)   
    program["eye_position"].write(eye)
    program["light"].write(light)
    modelObj.render()

# - Find the base center of the object.
# - Create the coordinates of the floor that is a square quad  parallel to the ZX plane and with sides 10 times the radius of the object.
#  - We will need this plane information (i.e. normal and the known point on the plane) to carry out our shadow projection in the shader. So save them in variables planePoint and planeNormal.
# - Write shader code to render this plane.
#   - So, create a texture sampler for the floor. Make provision in your shader code to render a texture
#
# Vertex shader(s)
#
_vertex_shader_code = '''
#version 460 core
in vec3 position;
in vec2 uv;

uniform mat4 view, perspective;
uniform vec3 normal;

out vec2 f_uv;
out vec3 f_position;
out vec3 f_normal;

void main() {
    f_position = position;
    f_normal = normalize(normal);
    f_uv = uv;
    gl_Position = perspective*view*vec4(position,1);
 }
'''
#
# Fragment shader(s)
#
_fragment_shader_code = '''
#version 460 core

uniform sampler2D map;
in vec2 f_uv;
in vec3 f_position;
in vec3 f_normal;

uniform vec4 light; 

const vec3 up = vec3(0, 1, 0);
const vec3 groundColor = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 skyColor = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);
    

// Add output variable here
out vec4 out_color;

void main() {
    vec3 L = normalize(light.xyz);
    if (light.w > 0) 
        L = normalize(light.xyz-f_position);
    vec3 N = normalize(f_normal);
    
    float w = dot(N,up);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 ambientColor = 0.1*(w*skyColor+(1-w)*groundColor)*materialColor;

    vec3 color = ambientColor + materialColor*clamp(dot(N,L),0,1);
    out_color =  vec4(color,1);
}
'''
floor_program = gl.program(
    vertex_shader= _vertex_shader_code,
    fragment_shader= _fragment_shader_code
)

# Base Plane Center of the bounding box:
# The extent of base plane parallel to XY plane
_minP = bound.boundingBox[0]
_maxP = glm.vec3(bound.boundingBox[1].x, _minP.y,bound.boundingBox[1].z) 
_center = (_minP + _maxP)/2

planePoint = _center
planeNormal = glm.vec3(0,1,0)

squareQuadSide = 3*bound.radius
halfSideSize = squareQuadSide/2
baseQuadGeomBuffer = gl.buffer(numpy.array([
    _center.x - halfSideSize, _center.y, _center.z - halfSideSize, 0, 0,
    _center.x + halfSideSize, _center.y, _center.z - halfSideSize, 1, 0,
    _center.x + halfSideSize, _center.y, _center.z + halfSideSize, 1, 1,
    _center.x - halfSideSize, _center.y, _center.z + halfSideSize, 0, 1
]).astype("float32"))
_index = numpy.array([
    0, 1, 2,
    2, 3, 0
]).astype("int32")
_indexBuffer = gl.buffer(_index)
floorRenderer = gl.vertex_array(
    floor_program,
    [(baseQuadGeomBuffer, "3f 2f", "position", "uv")],
    _indexBuffer,index_element_size=4
)
_texture_img = pygame.image.load("tile-squares-texture.jpg") 
_texture_data = pygame.image.tobytes(_texture_img,"RGB", True) 
_texture = gl.texture(_texture_img.get_size(), data = _texture_data, components=3)
_texture.build_mipmaps()
floor_sampler = gl.sampler(texture=_texture, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x = True, repeat_y = True)

def render_floor(view, perspective, light):
    floor_program["view"].write(view)
    floor_program["perspective"].write(perspective)
    floor_program["light"].write(light)
    floor_program["normal"].write(planeNormal)
    floor_sampler.use(0)
    floor_program["map"] = 0 
    floorRenderer.render()

displacement_vector = 4*bound.radius*glm.rotateX(glm.vec3(0,1,0), glm.radians(85)) #glm.vec3(0,0,1) 

light_displacement_vector = 4*bound.radius*glm.rotateZ(glm.vec3(0,1,0), glm.radians(45)) 
    
target_point = glm.vec3(bound.center)
up_vector = glm.vec3(0,1,0)

### View volume parameters
fov_radian = glm.radians(30) # In radian
aspect = width/height
near = bound.radius
far = 20*bound.radius
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

running = True
clock = pygame.time.Clock()
alpha = 0
lightAngle = 0


pause = True   # Keyboard key "p" toggles pause/orbit
pointLight = False # Keyboard key "l" toggles pause/orbit
shadow = False # Keyboard key "s" toggles shadow/noShadow
blend = False  # Keyboard key "b" toggles blend/no blend
print(" Camera Orbiting Paused. No Shadow. Directional Light. Blending Disabled.")

gl.enable(gl.DEPTH_TEST)

while running:   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif (event.type ==  pygame.KEYDOWN):
            if  event.key == 27:
                running = False
            elif event.key == pygame.K_p:
                pause = not pause
                print("Camera Orbiting" + " paused" if pause else "")
            elif event.key == pygame.K_s:
                shadow = not shadow
                print("shadow" if shadow else "No Shadow")
            elif event.key == pygame.K_b:
                blend = not blend
                print("blend" if blend else "No Blend")
            elif event.key == pygame.K_l:
                pointLight = not pointLight;
                print(("Point" if pointLight else "Directional") + " Light")
            elif event.key == pygame.K_LEFT:
                lightAngle -= 5
            elif event.key == pygame.K_RIGHT:
                lightAngle += 5
        elif (event.type == pygame.WINDOWRESIZED):
            width = event.x
            height = event.y
            perspectiveMatrix = glm.perspective(fov_radian, width/height, near, far)

    
    # create the aspect ratio correction matrix
    new_displacement_vector = glm.rotate(displacement_vector, glm.radians(alpha), glm.vec3(0,1,0))

    new_light_displacement_vector = glm.rotate(light_displacement_vector, glm.radians(lightAngle), glm.vec3(0,1,0))
    
    if pointLight:
        light = glm.vec4(target_point+new_light_displacement_vector,1)
    else:
        light = glm.vec4(new_light_displacement_vector,0)
        
    eye_point = target_point + new_displacement_vector

    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    gl.clear(0.,0.,0.)

    render_model(viewMatrix, perspectiveMatrix, light, eye_point)
    render_floor(viewMatrix, perspectiveMatrix, light)

    pygame.display.flip()
    clock.tick(60)  # limits FPS to 10
    if not pause:
        alpha +=  1
        if alpha > 360:
            alpha = 0
            
pygame.display.quit()