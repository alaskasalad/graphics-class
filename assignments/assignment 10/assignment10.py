import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V3 import create3DAssimpObject

size = 20
floor_positions = numpy.array([
    [-size/2, 0.0, -size/2],
    [ size/2, 0.0, -size/2],
    [ size/2, 0.0,  size/2],

    [ size/2, 0.0,  size/2],
    [-size/2, 0.0,  size/2],
    [-size/2, 0.0, -size/2],
], dtype="float32")

floor_uv = numpy.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],

    [1.0, 1.0],
    [0.0, 1.0],
    [0.0, 0.0],
], dtype="float32")

floor_geom = numpy.concatenate((floor_positions, floor_uv), axis=1).flatten().astype("float32")

class bound:
    boundingBox = [glm.vec3(-10.0, 0.0, -10.0), glm.vec3(10.0, 0.0, 10.0)]
    center = glm.vec3(0.0, 0.0, 0.0)
    radius = 10.0 * glm.sqrt(2.0)

rng = numpy.random.default_rng()

nSubDivisions = 10
totalInstances = nSubDivisions * nSubDivisions
delta = size / nSubDivisions
centers = []
minP = bound.boundingBox[0]

for i in range(nSubDivisions):
    for j in range(nSubDivisions):
        dxdy = rng.uniform(0, delta/2, 2)
        center = minP + glm.vec3([i*delta + delta/2, 0, j*delta + delta/2])
        centers.append(center)

objSize = delta * 0.5  # chairs will be 50% of their cell
scale_factor = 0.5 * delta 

#
# Programs
#

floor_vertex_shader= '''
    #version 430 core
    layout (location=0) in vec3 position;
    layout (location=1) in vec2 uv;
        
    uniform mat4 view, perspective, M;
    
    out vec2 f_uv; 
    out vec3 f_normal; 
    out vec3 f_position; 
    void main() {
        f_uv = uv;
        vec4 P = M*vec4(position, 1);
        f_position = P.xyz; 
        gl_Position = perspective*view*P;
        f_normal = vec3(0,1,0);
    }
    '''
floor_fragment_shader= '''
    #version 430 core
    in vec2 f_uv;
    in vec3 f_normal;
    
    uniform sampler2D map;
    uniform vec3 light;
    
    uniform float shininess;
    uniform vec3 eye_position;
    uniform vec3 k_diffuse;
    
    out vec4 out_color;
    
    vec3 computeColor(){
        vec3 L = normalize(light.xyz);
        vec3 materialColor = texture(map, f_uv).rgb;
        vec3 N = normalize(f_normal);
        float NdotL = dot(N,L);
        vec3 color = vec3(0.);
        vec3 ambientColor = 0.1 * materialColor;
        if (NdotL > 0.){
            vec3 diffuseColor = materialColor * NdotL;
            color = k_diffuse * diffuseColor;
        }
        color += ambientColor;
        return color;
    }
    void main() {
        out_color = vec4(computeColor(), 1);
    }
    '''    

model_file = "chair_table_class/scene.gltf"
modelObj = create3DAssimpObject(model_file)

#
# Programs
#

model_vertex_shader= '''
    #version 430 core
    layout (location=0) in vec3 position;
    layout (location=1) in vec3 normal;
    layout (location=2) in vec2 uv;

    layout(binding = 0, std430) readonly buffer InstanceData {
        mat4 instanceMatrix[];
    };
        
    uniform mat4 model, view, perspective;
    
    out vec2 f_uv; 
    out vec3 f_normal; 
    out vec3 f_position; 
    void main() {
        mat4 M = instanceMatrix[gl_InstanceID] * model;
        f_uv = uv;
        vec4 P = M*vec4(position, 1);
        f_position = P.xyz;
        gl_Position = perspective*view*P;
        mat3 normalMatrix = mat3(transpose(inverse(M)));
        f_normal = normalize(normalMatrix*normal);
    }
    '''
model_fragment_shader= '''
    #version 430 core
    in vec2 f_uv;
    in vec3 f_normal;
    in vec3 f_position;
    
    uniform sampler2D map;
    uniform vec3 light;
    
    uniform float shininess;
    uniform vec3 eye_position;
    uniform vec3 k_diffuse;
    
    out vec4 out_color;
    
    vec3 computeColor(){
        vec3 L = normalize(light.xyz);
        vec3 materialColor = texture(map, f_uv).rgb;
        vec3 N = normalize(f_normal);
        float NdotL = dot(N,L);
        vec3 color = vec3(0.);
        vec3 ambientColor = 0.1*materialColor;
        if (NdotL>0.){
            vec3 diffuseColor = materialColor * NdotL;
            vec3 V = normalize(eye_position - f_position);
            vec3 H = normalize(L+V);
            vec3 specularColor = vec3(0);
            if (shininess > 0)
                specularColor = vec3(pow(max(dot(N,H),0.0), shininess));
            color = k_diffuse * diffuseColor + specularColor;
        }
        color += ambientColor; 
        return color;
    }
    void main() {
        out_color = vec4(computeColor(), 1);
    }
    '''    

width = 840
height = 480

displacement_vector = 2*bound.radius*glm.rotate(glm.vec3(0,1,0), glm.radians(60), glm.vec3(1,0,0)) 
light_displacement_vector = 2*bound.radius*glm.rotate(glm.vec3(0,1,0), glm.radians(45), glm.vec3(1,0,0)) 
    
target_point = glm.vec3(bound.center)
up_vector = glm.vec3(0,1,0)

### View volume parameters
fov_radian = glm.radians(30) 
aspect = width/height
near = bound.radius
far = 3*bound.radius
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

pygame.init() 
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE) 
pygame.display.set_mode((width, height), flags= pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption(title = "Class Practice: Instructor")
gl = moderngl.get_context() 
gl.info["GL_VERSION"]
print(modelObj.bound.__dict__)

model_program = gl.program(model_vertex_shader, model_fragment_shader)
modelObj.createRenderableAndSampler(model_program)
floor_program = gl.program(floor_vertex_shader, floor_fragment_shader)

floor_renderable = gl.vertex_array(floor_program, 
                                   [(gl.buffer(floor_geom), "3f 2f", "position", "uv")]
                                )

texture_img = pygame.image.load("floor-wood.jpg")
texture_data = pygame.image.tobytes(texture_img, "RGB", True)
floor_texture = gl.texture(texture_img.get_size(), data = texture_data, components=3)
floor_sampler = gl.sampler(texture=floor_texture)

# === Compute base-center translation + scaling for chair ===
# If SceneBound has .min and .max (common pattern)
min_corner, max_corner = modelObj.bound.boundingBox
# base_center = middle of X and Z, bottom on Y
base_center = glm.vec3(
    (min_corner.x + max_corner.x) / 2,
    min_corner.y,
    (min_corner.z + max_corner.z) / 2
)

width = max_corner.x - min_corner.x
depth = max_corner.z - min_corner.z
max_dim = max(width, depth)

padding = 0.2  # tweak this
scale_factor = (delta * (1 - padding)) / max_dim

translation_matrix = glm.translate(glm.mat4(1.0), -base_center)
scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(scale_factor))
model_transform = scale_matrix * translation_matrix

matrixList = []
for center in centers:
    matrixList.append(list(glm.translate(center)))

instanceBuffer = gl.buffer(numpy.array(matrixList).astype("float32"))
instanceBuffer.bind_to_storage_buffer(0)

running = True
clock = pygame.time.Clock()
alpha = 0
pause = True
lightAngle = 0

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
            elif event.key == pygame.K_LEFT:
                lightAngle -= 5
            elif event.key == pygame.K_RIGHT:
                lightAngle += 5
        elif (event.type == pygame.WINDOWRESIZED):
            width = event.x
            height = event.y
            perspectiveMatrix = glm.perspective(fov_radian, width/height, near, far)

    new_displacement_vector = glm.rotate(displacement_vector, glm.radians(alpha), glm.vec3(0,1,0))
    new_light_displacement_vector = glm.rotate(light_displacement_vector, glm.radians(lightAngle), glm.vec3(0,1,0))
    eye_point = target_point + new_displacement_vector
    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    gl.clear(0.0, 0.0, 0.0)

    program = floor_program
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)
    program["M"].write(glm.mat4(1.0))  # identity for floor
    program["light"].write(new_light_displacement_vector) 
    floor_sampler.use(0)
    program["map"] = 0
    floor_renderable.render()

    program = modelObj.program
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)
    program["light"].write(new_light_displacement_vector)
    program["model"].write(model_transform)  # send translation + scaling
    modelObj.render(nInstances=totalInstances)
    
    pygame.display.flip()
    
    clock.tick(60)  
    if not pause:
        alpha +=  1
        if alpha > 360:
            alpha = 0
pygame.display.quit()
