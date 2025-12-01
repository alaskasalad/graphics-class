import glm
import numpy as np
import pygame
import moderngl
from math import cos, sin, sqrt
import numpy
from loadModelUsingAssimp_V3 import create3DAssimpObject
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

width = 840
height = 480

pygame.init()  # Initlizes its different modules. Display module is one of them.
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)  # pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption(title="Project Sub 3: Caitlin Box")
gl = moderngl.get_context()  # Get previously created context.
gl.info["GL_VERSION"]
FB = gl.detect_framebuffer()

modelFile = "mario_obj/scene.gltf"
modelObj = create3DAssimpObject(modelFile, verbose=False, textureFlag=True, normalFlag=True)
bound = modelObj.bound

# Base Plane Center of the bounding box:
# The extent of base plane parallel to XY plane
_minP = bound.boundingBox[0]
_maxP = glm.vec3(bound.boundingBox[1].x, _minP.y, bound.boundingBox[1].z)
_center = (_minP + _maxP) / 2

planePoint = _center
planeNormal = glm.vec3(0, 1, 0)

squareQuadSide = 3 * bound.radius
halfSideSize = squareQuadSide / 2
baseQuadGeomBuffer = gl.buffer(numpy.array([
    _center.x - halfSideSize, _center.y, _center.z - halfSideSize, 0, 1, 0, 0, 0,
    _center.x + halfSideSize, _center.y, _center.z - halfSideSize, 0, 1, 0, 1, 0,
    _center.x + halfSideSize, _center.y, _center.z + halfSideSize, 0, 1, 0, 1, 1,
    _center.x - halfSideSize, _center.y, _center.z + halfSideSize, 0, 1, 0, 0, 1
]).astype("float32"))
_index = numpy.array([
    0, 1, 2,
    2, 3, 0
]).astype("int32")
baseQuadIndexBuffer = gl.buffer(_index)

_wood_texture_img = pygame.image.load("tile-squares-texture.jpg")
_texture_data = pygame.image.tobytes(_wood_texture_img, "RGB", True)
_texture = gl.texture(_wood_texture_img.get_size(), data=_texture_data, components=3)
_texture.build_mipmaps()
floor_sampler = gl.sampler(texture=_texture, filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR), repeat_x=True, repeat_y=True)

def queryProgram(program):
    for name in program:
        member = program[name]
        print(name, type(member), member)

#
# Vertex shader(s)
#
vertex_shader_code = '''
#version 460 core
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;

uniform mat4 model, view, perspective;

// light-space matrices for shadow mapping
uniform mat4 lightViewMatrix;
uniform mat4 lightProjectionMatrix;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;
out vec4 f_lightSpace;

void main() {
    vec4 P = model * vec4(position, 1.0);
    f_position = P.xyz;

    f_uv = uv;
    mat3 normalMatrix = mat3(transpose(inverse(model))); // inverse transpose of model transformation
    f_normal = normalize(normalMatrix * normal);

    // position for main camera
    gl_Position = perspective * view * P;

    // position in light space (for shadow mapping)
    f_lightSpace = lightProjectionMatrix * lightViewMatrix * P;
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
in vec4 f_lightSpace;

uniform sampler2D map;

// Shadow mapping uniforms
uniform sampler2D shadowMap;
uniform bool biasFlag;
uniform int pcf;
uniform bool useShadowMap;

uniform vec3 light;

uniform float shininess;
uniform vec3 eye_position;
uniform vec3 k_diffuse;

const vec3 up = vec3(0, 1, 0);
const vec3 groundColor = vec3(0.3215686274509804, 0.4, 0.10980392156862745);
const vec3 skyColor = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

out vec4 out_color;

//===============================
// Depth fetch with bias (NOTES)
// depthStored = texture(shadowMap, uv).r + bias
//===============================
float getDepthStored(vec2 uv) {
    float depthStored = texture(shadowMap, uv).r;
    if (biasFlag) {
        depthStored += 0.001;   // bias > 0: reduces acne, softens jaggies
    }
    return depthStored;
}

//===============================
// Binary visibility (NOTES)
// return (depthStored < currentDepth) ? 0 : 1;
//===============================
float computeBinaryVisibility(vec2 shadowUV, float currentDepth) {
    float depthStored = getDepthStored(shadowUV);
    return (depthStored < currentDepth) ? 0.0 : 1.0;
}

//===============================
// PCF visibility (fractional 0..1)
// Still uses depthStored + bias < currentDepth
//===============================
float computePCFVisibility(vec2 shadowUV, float currentDepth) {
    int kernelRadius = 1;
    if (pcf == 2) kernelRadius = 2;
    else if (pcf == 3) kernelRadius = 3;

    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);

    float visibility = 0.0;
    int samples = 0;

    for (int x = -kernelRadius; x <= kernelRadius; x++) {
        for (int y = -kernelRadius; y <= kernelRadius; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            float depthStored = getDepthStored(shadowUV + offset);

            // If depthStored + bias < currentDepth → in shadow → 0
            visibility += (depthStored < currentDepth) ? 0.0 : 1.0;
            samples++;
        }
    }
    return visibility / float(samples);
}

//===============================
// Main visibility factor
//===============================
float ComputeVisibilityFactor() {
    if (!useShadowMap) return 1.0;

    // Light-space projection → UV + depth
    vec3 P_Light = f_lightSpace.xyz / f_lightSpace.w;
    float currentDepth = P_Light.z * 0.5 + 0.5;
    vec2 shadowUV = P_Light.xy * 0.5 + 0.5;

    // Outside shadow map → fully lit
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0) {
        return 1.0;
    }

    // No PCF → binary
    if (pcf == 0) {
        return computeBinaryVisibility(shadowUV, currentDepth);
    }

    // PCF → fractional
    return computePCFVisibility(shadowUV, currentDepth);
}

vec3 computeColor() {
    vec3 L = normalize(light.xyz - f_position);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal);
    float NdotL = dot(N, L);
    vec3 color = vec3(0.0);

    float w = dot(N, up);
    vec3 ambientColor = 0.25 * (w * skyColor + (1.0 - w) * groundColor) * materialColor;

    float fractionalLightVisibility = ComputeVisibilityFactor();

    if (NdotL > 0.0) {
        vec3 diffuselyReflectedColor = materialColor * NdotL;
        // Compute specular color
        vec3 V = normalize(eye_position - f_position);
        vec3 H = normalize(L + V);
        vec3 specularlyReflectedColor = vec3(0.0);
        if (shininess > 0.0)
            specularlyReflectedColor = vec3(1.0) * pow(max(dot(N, H), 0.0), shininess);
        color = fractionalLightVisibility * (k_diffuse * diffuselyReflectedColor + specularlyReflectedColor);
    }
    color += ambientColor;
    return color;
}

void main() {
    out_color = vec4(computeColor(), 1.0);
}
'''

#
# Programs
#
program = gl.program(
    vertex_shader=vertex_shader_code,
    fragment_shader=fragment_shader_code
)

queryProgram(program)

# Model renderables & samplers for main program
modelObj.createRenderableAndSampler(program)

floorRenderer = gl.vertex_array(
    program,
    [(baseQuadGeomBuffer, "3f 3f 2f", "position", "normal", "uv")],
    baseQuadIndexBuffer,
    index_element_size=4
)

SHADOW_SIZE = (2048, 2048)
SHADOW_TEX_UNIT = 1  # use texture unit 1 for shadow map (0 is used by color textures)

# Create depth texture
depthBuffer = gl.depth_texture(SHADOW_SIZE)
depthBuffer.compare_func = '<'

# Create sampler for shadow map
shadowMapSampler = gl.sampler(
    texture=depthBuffer,
    filter=(gl.LINEAR, gl.LINEAR),
    repeat_x=False,
    repeat_y=False
)

# Create framebuffer with only depth attachment
shadow_fbo = gl.framebuffer(
    depth_attachment=depthBuffer
)

shadowmap_vertex_shader = '''
#version 460 core
layout (location=0) in vec3 position;
layout (location=1) in vec2 uv;

out vec2 f_uv;

void main() {
    f_uv = uv;
    gl_Position = vec4(position, 1.0);
}
'''

shadowmap_fragment_shader = '''
#version 460 core
in vec2 f_uv;
uniform sampler2D shadowMap;
out vec4 out_color;

void main() {
    float d = texture(shadowMap, f_uv).r;
    // Show depth as grayscale
    out_color = vec4(vec3(d), 1.0);
}
'''

shadowmap_program = gl.program(
    vertex_shader=shadowmap_vertex_shader,
    fragment_shader=shadowmap_fragment_shader
)

shadow_quad_data = np.array([
    #  position (clip space z = 0)          uv
    -1.0, -1.0, 0.0,                        0.0, 0.0,
     1.0, -1.0, 0.0,                        1.0, 0.0,
     1.0,  1.0, 0.0,                        1.0, 1.0,
    -1.0,  1.0, 0.0,                        0.0, 1.0,
], dtype='f4')

shadow_quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype='i4')

shadow_quad_buffer = gl.buffer(shadow_quad_data)
shadow_quad_index_buffer = gl.buffer(shadow_quad_indices)

shadow_quad_vao = gl.vertex_array(
    shadowmap_program,
    [(shadow_quad_buffer, "3f 2f", "position", "uv")],
    shadow_quad_index_buffer,
    index_element_size=4
)

def render_model():
    modelObj.render()


def render_floor():
    floor_sampler.use(0)
    program["model"].write(glm.mat4(1))
    floorRenderer.render()


def renderScene(viewMatrix, perspectiveMatrix, light, eye):
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)
    program["eye_position"].write(eye)
    program["light"].write(light)
    render_model()
    program["shininess"] = 0
    render_floor()

def showShadowMap():
    size = width // 4
    # place viewport in top-right corner
    gl.viewport = (width - size, height - size, size, size)
    gl.clear(0.5, 0.0, 0.5, viewport=gl.viewport)
    shadowMapSampler.use(SHADOW_TEX_UNIT)
    shadowmap_program["shadowMap"] = SHADOW_TEX_UNIT
    shadow_quad_vao.render()
    # restore full window viewport
    gl.viewport = (0, 0, width, height)

displacement_vector = 4 * bound.radius * glm.rotateX(glm.vec3(0, 1, 0), glm.radians(85))

light_displacement_vector = 4 * bound.radius * glm.rotateZ(glm.vec3(0, 1, 0), glm.radians(45))

target_point = glm.vec3(bound.center)
up_vector = glm.vec3(0, 1, 0)

fov_radian = glm.radians(30)  # In radian
aspect = width / height
near = bound.radius
far = 20 * bound.radius
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

running = True
clock = pygame.time.Clock()
alpha = 0
lightAngle = 0
pcf = 0

pause = True   # Keyboard key "p" toggles pause/orbit
debug = False  # Keyboard key "d" toggles display of shadowmap
bias = True    # Keyboard key "b" toggles bias/no bias
print(" Camera Orbiting Paused. No Shadow. Point Light. Bias. PCF : ", pcf)
gl.enable(gl.DEPTH_TEST)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif (event.type == pygame.KEYDOWN):
            if event.key == 27:
                running = False
            elif event.key == pygame.K_p:
                pause = not pause
            elif event.key == pygame.K_d:
                debug = not debug
            elif event.key == pygame.K_b:
                bias = not bias
            elif event.key == pygame.K_LEFT:
                lightAngle -= 5
            elif event.key == pygame.K_RIGHT:
                lightAngle += 5
            elif event.key == pygame.K_UP:
                if pcf < 3:
                    pcf += 1
                print("PCF level: ", pcf)
            elif event.key == pygame.K_DOWN:
                if pcf > 0:
                    pcf -= 1
                print("PCF level: ", pcf)
        elif (event.type == pygame.WINDOWRESIZED):
            width = event.x
            height = event.y
            perspectiveMatrix = glm.perspective(fov_radian, width / height, near, far)
            gl.viewport = (0, 0, width, height)

    # camera orbit
    new_displacement_vector = glm.rotateY(displacement_vector, glm.radians(alpha))
    new_light_displacement_vector = glm.rotateY(light_displacement_vector, glm.radians(lightAngle))

    light_point = target_point + new_light_displacement_vector
    eye_point = target_point + new_displacement_vector

    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    # light camera matrices for shadow mapping (Pass 1 + Pass 2)
    lightViewMatrix = glm.lookAt(light_point, target_point, up_vector)
    lightPerspectiveMatrix = glm.perspective(glm.radians(60.0), 1.0, bound.radius, 10.0 * bound.radius)

    # upload shadow matrices
    program["lightViewMatrix"].write(lightViewMatrix)
    program["lightProjectionMatrix"].write(lightPerspectiveMatrix)

    # update bias / pcf uniforms
    program["biasFlag"].value = bias
    program["pcf"].value = pcf

    # pass 1
    shadow_fbo.use()
    gl.viewport = (0, 0, SHADOW_SIZE[0], SHADOW_SIZE[1])
    shadow_fbo.clear(depth=1.0)
    program["useShadowMap"].value = False

    # render scene from light's pov
    renderScene(lightViewMatrix, lightPerspectiveMatrix, light_point, light_point)

    # pass 2
    FB.use()
    gl.viewport = (0, 0, width, height)
    gl.clear(0.2, 0.2, 0.0)
    program["useShadowMap"].value = True
    # bind shadow map sampler
    shadowMapSampler.use(SHADOW_TEX_UNIT)
    program["shadowMap"] = SHADOW_TEX_UNIT

    # render scene from camera
    renderScene(viewMatrix, perspectiveMatrix, light_point, eye_point)

    # show shadow map as small viewport
    if debug:
        showShadowMap()

    pygame.display.flip()
    clock.tick(60)  # limits FPS to 60
    if not pause:
        alpha += 1
        if alpha > 360:
            alpha = 0

pygame.display.quit()
