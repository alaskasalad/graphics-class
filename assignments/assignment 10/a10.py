import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V3 import create3DAssimpObject

# =========================
# Scene & Floor Parameters
# =========================
size = 20.0                 # 20 units on a side (assignment requirement)
nSubDivisions = 10
totalInstances = nSubDivisions * nSubDivisions
delta = size / nSubDivisions

# ---- Square floor on XZ plane, centered at origin (Y is up) ----
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

# ======= Bound for camera & light per assignment =======
class bound:
    boundingBox = [glm.vec3(-10.0, 0.0, -10.0), glm.vec3(10.0, 0.0, 10.0)]
    center = glm.vec3(0.0, 0.0, 0.0)
    radius = 10.0 * glm.sqrt(2.0)

rng = numpy.random.default_rng()

# =========================
# Model & Shaders
# =========================
model_file = "chair_table_class/scene.gltf"
modelObj = create3DAssimpObject(model_file)

# ---- Floor shaders ----
floor_vertex_shader = '''
#version 430 core
layout (location=0) in vec3 position;
layout (location=1) in vec2 uv;

uniform mat4 view, perspective, M;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;

void main() {
    f_uv = uv;
    vec4 P = M * vec4(position, 1.0);
    f_position = P.xyz;
    gl_Position = perspective * view * P;
    f_normal = vec3(0.0, 1.0, 0.0);  // Upwards normal for XZ floor
}
'''

floor_fragment_shader = '''
#version 430 core
in vec2 f_uv;
in vec3 f_normal;
in vec3 f_position;

uniform sampler2D map;
uniform vec3 light;

out vec4 out_color;

vec3 computeColor(){
    vec3 L = normalize(light);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal);
    float NdotL = max(dot(N, L), 0.0);
    vec3 ambient = 0.1 * materialColor;
    vec3 color = ambient;
    if (NdotL > 0.0) {
        vec3 diffuse = materialColor * NdotL;
        color += diffuse;
    }
    return color;
}

void main() {
    out_color = vec4(computeColor(), 1.0);
}
'''

# ---- Model shaders with SSBO for instance matrices ----
model_vertex_shader = '''
#version 430 core
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;

layout(binding = 0, std430) readonly buffer InstanceData {
    mat4 instanceMatrix[];
};

uniform mat4 model;        // global model (pre-transform like base-center removal)
uniform mat4 view, perspective;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;

void main() {
    mat4 M = instanceMatrix[gl_InstanceID] * model;
    f_uv = uv;
    vec4 P = M * vec4(position, 1.0);
    f_position = P.xyz;
    gl_Position = perspective * view * P;

    mat3 normalMatrix = mat3(transpose(inverse(M)));
    f_normal = normalize(normalMatrix * normal);
}
'''

model_fragment_shader = '''
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
    vec3 L = normalize(light);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal);
    float NdotL = max(dot(N, L), 0.0);
    vec3 color = vec3(0.0);
    vec3 ambient = 0.1 * materialColor;

    if (NdotL > 0.0){
        vec3 diffuse = materialColor * NdotL;
        vec3 V = normalize(eye_position - f_position);
        vec3 H = normalize(L + V);
        vec3 specular = vec3(0.0);
        if (shininess > 0.0)
            specular = vec3(pow(max(dot(N, H), 0.0), shininess));
        color = k_diffuse * diffuse + specular;
    }
    color += ambient;
    return color;
}

void main() {
    out_color = vec4(computeColor(), 1.0);
}
'''

# =========================
# Window / GL setup
# =========================
width, height = 960, 540

pygame.init()
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption("Assignment 10: Caitlin Box")

gl = moderngl.get_context()
_ = gl.info.get("GL_VERSION")

# Programs
model_program = gl.program(vertex_shader=model_vertex_shader, fragment_shader=model_fragment_shader)
modelObj.createRenderableAndSampler(model_program)

floor_program = gl.program(vertex_shader=floor_vertex_shader, fragment_shader=floor_fragment_shader)

# Floor renderable
floor_renderable = gl.vertex_array(
    floor_program,
    [(gl.buffer(floor_geom), "3f 2f", "position", "uv")]
)

# Floor texture
texture_img = pygame.image.load("floor-wood.jpg")
texture_data = pygame.image.tobytes(texture_img, "RGB", True)
floor_texture = gl.texture(texture_img.get_size(), components=3, data=texture_data)
floor_sampler = gl.sampler(texture=floor_texture)

# =========================
# Compute per-instance transforms
# =========================
min_corner, max_corner = modelObj.bound.boundingBox

# base center = mid XZ at bottom Y
base_center = glm.vec3(
    (min_corner.x + max_corner.x) / 2.0,
    min_corner.y,
    (min_corner.z + max_corner.z) / 2.0
)
T_base_to_origin = glm.translate(glm.mat4(1.0), -base_center)

# scale model to 75% of cell
width_x = max_corner.x - min_corner.x
depth_z = max_corner.z - min_corner.z
max_footprint = max(width_x, depth_z)
cell_target = 0.75 * delta
scale_factor = (cell_target / max_footprint) if max_footprint > 0 else 1.0
S_fit = glm.scale(glm.mat4(1.0), glm.vec3(scale_factor))

pre_model = S_fit * T_base_to_origin

matrixList = []
minP = glm.vec3(-size/2.0, 0.0, -size/2.0)

for i in range(nSubDivisions):
    for j in range(nSubDivisions):
        # base cell center
        base_center_cell = minP + glm.vec3(i*delta + delta/2.0, 0.0, j*delta + delta/2.0)
        jitter = rng.uniform(-0.2*delta, 0.2*delta, 2)
        cell_center = base_center_cell + glm.vec3(jitter[0], 0.0, jitter[1])

        angle_deg = rng.uniform(-30.0, 30.0)
        R_y = glm.rotate(glm.mat4(1.0), glm.radians(angle_deg), glm.vec3(0,1,0))
        T_cell = glm.translate(glm.mat4(1.0), cell_center)

        M_instance = T_cell * R_y * pre_model
        matrixList.append([elem for col in M_instance for elem in col])

instanceBuffer = gl.buffer(numpy.array(matrixList, dtype="float32"))
instanceBuffer.bind_to_storage_buffer(0)

# =========================
# Camera & Light
# =========================
init_dir = glm.normalize(glm.vec3(20.0, 10.0, 0.0))
orbit_radius = 25.0
alpha_deg = 0.0

fov_radian = glm.radians(45.0)
near, far = 0.1, 200.0
aspect = width / height
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

# ================
# Render loop
# ================
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

    rotY = glm.rotate(glm.mat4(1.0), glm.radians(alpha_deg), glm.vec3(0, 1, 0))
    cam_dir = glm.vec3(rotY * glm.vec4(init_dir, 0.0))
    eye_point = cam_dir * orbit_radius
    target_point = glm.vec3(0.0, 0.0, 0.0)
    up_vector = glm.vec3(0.0, 1.0, 0.0)
    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    rotY_light = glm.rotate(glm.mat4(1.0), glm.radians(lightAngle), glm.vec3(0, 1, 0))
    light_dir = glm.normalize(glm.vec3(rotY_light * glm.vec4(1.0, 1.0, 0.5, 0.0)))

    gl.clear(0.05, 0.05, 0.07)

    # ---- Draw floor ----
    fp = floor_program
    fp["view"].write(viewMatrix)
    fp["perspective"].write(perspectiveMatrix)
    fp["M"].write(glm.mat4(1.0))
    fp["light"].write(light_dir)
    floor_sampler.use(0)
    fp["map"] = 0
    floor_renderable.render()

    # ---- Draw models ----
    mp = modelObj.program
    mp["view"].write(viewMatrix)
    mp["perspective"].write(perspectiveMatrix)
    mp["light"].write(light_dir)
    mp["shininess"].value = 32.0
    mp["k_diffuse"].write(glm.vec3(1.0, 1.0, 1.0))
    mp["eye_position"].write(eye_point)
    mp["model"].write(glm.mat4(1.0))
    modelObj.render(nInstances=totalInstances)

    pygame.display.flip()
    clock.tick(60)

    if not pause:
        alpha_deg = (alpha_deg + 30.0 * (1.0 / 60.0)) % 360.0

pygame.display.quit()
