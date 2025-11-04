# imports
import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V2 import create3DAssimpObject

# Initialize window
width, height = 840, 480
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption("Utah Teapot Viewer")
gl = moderngl.get_context()

# ============================
# Vertex shader
# ============================
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
    vec4 P = model * vec4(position, 1.0);
    f_position = P.xyz;
    gl_Position = perspective * view * P;
    f_normal = normalize(normalMatrix * normal);
}
'''

# ============================
# Fragment shader
# ============================
fragment_shader_code = '''
#version 460 core

in vec3 f_normal;
in vec3 f_position;
in vec2 f_uv;

uniform sampler2D map;
uniform vec3 light;
uniform vec3 eye_position;
uniform bool metal;

const float shininess = 5.0;

out vec4 out_color;

vec3 computeColor() {
    vec3 color = vec3(0);
    vec3 N = normalize(f_normal);
    vec3 V = normalize(eye_position - f_position);
    vec3 L = normalize(light);
    vec3 H = normalize(L + V);
    vec3 materialColor = texture(map, f_uv).rgb;

    if (dot(N, L) > 0.0) {
        if (metal)
            color = materialColor * pow(dot(N, H), shininess);
        else
            color = materialColor * (0.3 + 0.7 * max(dot(N, L), 0.0));
    }
    return color;
}

void main() {
    out_color = vec4(computeColor(), 1.0);
}
'''

# ============================
# Compile shaders
# ============================
model_program = gl.program(
    vertex_shader=vertex_shader_code,
    fragment_shader=fragment_shader_code
)
format = "3f 3f 2f"
variables = ["position", "normal", "uv"]

# ============================
# Load model
# ============================
modelFile = "the_utah_teapot/scene.gltf"
modelObj = create3DAssimpObject(modelFile, verbose=False, textureFlag=True, normalFlag=True)
model_renderables = modelObj.getRenderables(gl, model_program, format, variables)
scene = modelObj.scene
bound = modelObj.bound

# ============================
# Recursive render
# ============================
def recursive_render(node, M):
    nodeTransform = glm.transpose(glm.mat4(node.transformation))
    currentTransform = M * nodeTransform
    if node.num_meshes > 0:
        for index in node.mesh_indices:
            model_renderables[index]._program["model"].write(currentTransform)
            normalMatrix = glm.mat3(glm.transpose(glm.inverse(currentTransform)))
            model_renderables[index]._program["normalMatrix"].write(normalMatrix)
            model_renderables[index].render()
    for child in node.children:
        recursive_render(child, currentTransform)

def render():
    recursive_render(scene.root_node, M=glm.mat4(1))

# ============================
# Textures
# ============================
# Gold texture
_imageFile = "gold.jpg"
_texture_img = pygame.image.load(_imageFile)
_texture_data = pygame.image.tobytes(_texture_img, "RGB", True)
_texture = gl.texture(_texture_img.get_size(), data=_texture_data, components=3)
gold_sampler = gl.sampler(texture=_texture)

# Matte (gray) texture
gray_surface = pygame.Surface((2, 2))
gray_surface.fill((180, 180, 180))
gray_data = pygame.image.tobytes(gray_surface, "RGB", True)
gray_texture = gl.texture(gray_surface.get_size(), data=gray_data, components=3)
matte_sampler = gl.sampler(texture=gray_texture)

# ============================
# Skybox (simple gray background)
# ============================
_positions = numpy.array([
    [-1, 1],
    [ 1, 1],
    [ 1,-1],
    [-1,-1]
], dtype="float32")

_index = numpy.array([0, 1, 2, 2, 3, 0], dtype="int32")

_vertex_shader_code = '''
#version 460 core
in vec2 position;
void main() {
    gl_Position = vec4(position, 1.0, 1.0);
}
'''

_fragment_shader_code = '''
#version 460 core
out vec4 out_color;
void main() {
    out_color = vec4(0.5, 0.5, 0.5, 1.0);
}
'''

skybox_program = gl.program(
    vertex_shader=_vertex_shader_code,
    fragment_shader=_fragment_shader_code
)

skybox_renderable = gl.vertex_array(
    skybox_program,
    [(gl.buffer(_positions.flatten()), "2f", "position")],
    index_buffer=gl.buffer(_index), index_element_size=4
)

# ============================
# Camera setup
# ============================
displacement_vector = 2 * bound.radius * glm.rotate(glm.vec3(0, 1, 0), glm.radians(85), glm.vec3(1, 0, 0))
light_displacement_vector = 2 * bound.radius * glm.rotate(glm.vec3(0, 1, 0), glm.radians(45), glm.vec3(1, 0, 0))
target_point = glm.vec3(bound.center)
up_vector = glm.vec3(0, 1, 0)

# View volume
fov_radian = glm.radians(45)
aspect = width / height
near = bound.radius
far = 3 * bound.radius
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

# ============================
# Main loop
# ============================
running = True
clock = pygame.time.Clock()
alpha = 0
lightAngle = 0
pause = True
metal = False
skybox = False

gl.enable(gl.DEPTH_TEST)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == 27:
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
        elif event.type == pygame.WINDOWRESIZED:
            width = event.x
            height = event.y
            perspectiveMatrix = glm.perspective(fov_radian, width / height, near, far)

    new_displacement_vector = glm.rotate(displacement_vector, glm.radians(alpha), glm.vec3(0, 1, 0))
    new_light_displacement_vector = glm.rotate(light_displacement_vector, glm.radians(lightAngle), glm.vec3(0, 1, 0))
    eye_point = target_point + new_displacement_vector
    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    gl.clear(0.5, 0.5, 0.0)

    # Render model
    program = model_program
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)
    program["eye_position"].write(eye_point)
    program["light"].write(new_light_displacement_vector)
    program["map"] = 0
    program["metal"] = metal

    if metal:
        gold_sampler.use(0)
    else:
        matte_sampler.use(0)

    render()

    if skybox:
        skybox_renderable.render()

    pygame.display.flip()
    clock.tick(60)
    if not pause:
        alpha += 1
        if alpha > 360:
            alpha = 0

pygame.display.quit()
