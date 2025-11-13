import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V3 import create3DAssimpObject
from OpenGL.GL import *

width, height = 840, 480
pygame.init()

# GL attributes
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption("Project Substitute: Assignment 02 - Caitlin Box")

gl = moderngl.get_context()

# ---------------- VERTEX SHADER ----------------
vertex_shader_code = '''
#version 460 core
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;

uniform mat4 model, view, perspective;
uniform vec4 light;
uniform bool isShadow;
uniform vec3 planePoint;
uniform vec3 planeNormal;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;

void main() {
    vec4 worldPos4 = model * vec4(position, 1.0);
    vec3 worldPos = worldPos4.xyz;
    f_position = worldPos;
    f_uv = uv;
    f_normal = normalize(mat3(transpose(inverse(model))) * normal);

    if (!isShadow) {
        gl_Position = perspective * view * worldPos4;
    } else {
        // Shadow projection onto plane
        vec3 L;
        if (light.w > 0.0) {
            // Point light: ray from light → vertex
            L = light.xyz - worldPos;
        } else {
            // Directional light
            L = light.xyz;
        }

        float denom = dot(L, planeNormal);
        if (abs(denom) < 1e-5) {
            gl_Position = perspective * view * worldPos4;
        } else {
            float t = dot(planePoint - worldPos, planeNormal) / denom;
            vec3 shadowPos = worldPos + t * L + 0.005 * planeNormal;
            gl_Position = perspective * view * vec4(shadowPos, 1.0);
        }
    }
}
'''

# ---------------- FRAGMENT SHADER ----------------
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
uniform bool isShadow;

const vec3 up = vec3(0, 1, 0);
const vec3 groundColor = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 skyColor = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

out vec4 out_color;

vec3 computeColor(){
    vec3 L = normalize(light.xyz);
    if (light.w > 0)
        L = normalize(light.xyz - f_position);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal);
    float NdotL = dot(N, L);
    vec3 color = vec3(0.0);
    float w = dot(N, up);
    vec3 ambientColor = 0.25 * (w * skyColor + (1.0 - w) * groundColor) * materialColor;
    if (NdotL > 0.0){
        vec3 diffuselyReflectedColor = materialColor * NdotL;
        vec3 V = normalize(eye_position - f_position);
        vec3 H = normalize(L + V);
        vec3 specularlyReflectedColor = vec3(0.0);
        if (shininess > 0.0)
            specularlyReflectedColor = vec3(1.0) * pow(dot(N, H), shininess);
        color = k_diffuse * diffuselyReflectedColor + specularlyReflectedColor;
    }
    color += ambientColor; 
    return color;
}

void main() {
    if (isShadow) {
        out_color = vec4(0.1, 0.1, 0.1, 0.5);
    } else {
        out_color = vec4(computeColor(), 1.0);
    }
}
'''

# ---------------- MODEL PROGRAM ----------------
model_program = gl.program(vertex_shader=vertex_shader_code, fragment_shader=fragment_shader_code)
variables = ["position", "normal", "uv"]

modelFile = "mario_obj/scene.gltf"
modelObj = create3DAssimpObject(modelFile, verbose=False, textureFlag=True, normalFlag=True)
modelObj.createRenderableAndSampler(model_program)
bound = modelObj.bound

def render_model(view, perspective, light, eye):
    program = model_program
    program["view"].write(view)
    program["perspective"].write(perspective)
    program["eye_position"].write(eye)
    program["light"].write(light)
    program["isShadow"].value = False
    modelObj.render()

# ---------------- FLOOR ----------------
floor_vshader = '''
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
    gl_Position = perspective * view * vec4(position, 1.0);
}
'''

floor_fshader = '''
#version 460 core
uniform sampler2D map;
in vec2 f_uv;
in vec3 f_position;
in vec3 f_normal;
uniform vec4 light; 

const vec3 up = vec3(0, 1, 0);
const vec3 groundColor = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 skyColor = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);
out vec4 out_color;

void main() {
    vec3 L = normalize(light.xyz);
    if (light.w > 0.0) 
        L = normalize(light.xyz - f_position);
    vec3 N = normalize(f_normal);
    float w = dot(N, up);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 ambientColor = 0.1 * (w * skyColor + (1.0 - w) * groundColor) * materialColor;
    vec3 color = ambientColor + materialColor * clamp(dot(N, L), 0.0, 1.0);
    out_color = vec4(color, 1.0);
}
'''

floor_program = gl.program(vertex_shader=floor_vshader, fragment_shader=floor_fshader)

_minP = bound.boundingBox[0]
_maxP = glm.vec3(bound.boundingBox[1].x, _minP.y, bound.boundingBox[1].z)
_center = (_minP + _maxP) / 2
planePoint = _center
planeNormal = glm.vec3(0, 1, 0)

side = 3 * bound.radius
h = side / 2
vbo = gl.buffer(numpy.array([
    _center.x - h, _center.y, _center.z - h, 0, 0,
    _center.x + h, _center.y, _center.z - h, 1, 0,
    _center.x + h, _center.y, _center.z + h, 1, 1,
    _center.x - h, _center.y, _center.z + h, 0, 1
]).astype("float32"))
ibo = gl.buffer(numpy.array([0, 1, 2, 2, 3, 0]).astype("int32"))
floor = gl.vertex_array(floor_program, [(vbo, "3f 2f", "position", "uv")], ibo, index_element_size=4)

tex = pygame.image.load("tile-squares-texture.jpg")
tex_data = pygame.image.tobytes(tex, "RGB", True)
texture = gl.texture(tex.get_size(), data=tex_data, components=3)
texture.build_mipmaps()
sampler = gl.sampler(texture=texture, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)

def render_floor(view, perspective, light):
    floor_program["view"].write(view)
    floor_program["perspective"].write(perspective)
    floor_program["light"].write(light)
    floor_program["normal"].write(planeNormal)
    sampler.use(0)
    floor_program["map"] = 0
    floor.render()

# ---------------- CAMERA + LIGHT ----------------
disp_vec = 4 * bound.radius * glm.rotateX(glm.vec3(0, 1, 0), glm.radians(85))
light_disp = 4 * bound.radius * glm.rotateZ(glm.vec3(0, 1, 0), glm.radians(45))
target = glm.vec3(bound.center)
up = glm.vec3(0, 1, 0)
fov = glm.radians(30)
aspect = width / height
near = bound.radius
far = 20 * bound.radius
persp = glm.perspective(fov, aspect, near, far)

alpha = 0
angle = 0
pause = True
pointLight = False
shadow = False
blend = False

gl.enable(gl.DEPTH_TEST)
glClearColor(0.0, 0.0, 0.0, 1.0)

model_program["planePoint"].write(planePoint)
model_program["planeNormal"].write(planeNormal)
model_program["isShadow"].value = False

clock = pygame.time.Clock()
running = True

print("Directional Light | No Shadow | Blend OFF")

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == 27:
                running = False
            elif e.key == pygame.K_p:
                pause = not pause
            elif e.key == pygame.K_s:
                shadow = not shadow
            elif e.key == pygame.K_b:
                blend = not blend
            elif e.key == pygame.K_l:
                pointLight = not pointLight

    # Light
    if pointLight:
        light = glm.vec4(target + glm.rotate(light_disp, glm.radians(angle), glm.vec3(0, 1, 0)), 1.0)
    else:
        light = glm.vec4(glm.rotate(light_disp, glm.radians(angle), glm.vec3(0, 1, 0)), 0.0)

    # Camera
    eye = target + glm.rotate(disp_vec, glm.radians(alpha), glm.vec3(0, 1, 0))
    view = glm.lookAt(eye, target, up)

    # ----- PASS 1 -----
    if blend or pointLight:
        # Use stencil for blend (or for point light fix)
        glEnable(GL_STENCIL_TEST)
        glStencilMask(0xFF)
        glClearStencil(0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
    else:
        # Original no-stencil mode for directional light + no blend
        glDisable(GL_STENCIL_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDisable(GL_BLEND)
    render_model(view, persp, light, eye)
    render_floor(view, persp, light)

    # ----- PASS 2 -----
    if shadow:
        if blend:
            # Original soft blend behavior
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_STENCIL_TEST)
            glStencilFunc(GL_EQUAL, 1, 0xFF)
            glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO)
        elif pointLight:
            # No blend but keep stencil for floor lock (point light only)
            glDisable(GL_BLEND)
            glEnable(GL_STENCIL_TEST)
            glStencilFunc(GL_EQUAL, 1, 0xFF)
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        else:
            # Pure no-blend no-stencil for directional light
            glDisable(GL_BLEND)
            glDisable(GL_STENCIL_TEST)

        model_program["view"].write(view)
        model_program["perspective"].write(persp)
        model_program["eye_position"].write(eye)
        model_program["light"].write(light)
        model_program["isShadow"].value = True
        modelObj.render()
        model_program["isShadow"].value = False

    pygame.display.flip()
    clock.tick(60)
    if not pause:
        alpha += 1
        if alpha > 360:
            alpha = 0

pygame.display.quit()
