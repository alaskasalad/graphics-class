import pygame
import moderngl
import glm
from loadModelUsingAssimp_V2 import create3DAssimpObject

assimpObject = create3DAssimpObject("snoopy_flying_ace/scene.gltf", verbose=False, textureFlag = True, normalFlag= True) # changed mario to chair thing 
              # Flags supported: textureFlag, normalFlag, tangentFlag

#
# Vertex shader(s)
#
vertex_shader_code = '''
#version 460 core

layout (location=0) in vec3 position;
layout (location=1) in vec2 uv;
layout (location=2) in vec3 normal;

uniform mat4 model, view, perspective;
out vec2 f_uv; // Texture coordinate
out vec3 f_normal; // normal vector output to the pipeline 

void main() {
    f_uv = uv;
    vec4 P = perspective*view*model*vec4(position, 1);
    gl_Position = P;
    mat3 normalMatrix = mat3(transpose(inverse(model))); // inverse transpose of model transformation
    f_normal = normalize(normalMatrix*normal); 
}
'''

#
# Fragment shader(s)
#
fragment_shader_code = '''
#version 460 core

in vec2 f_uv;
in vec3 f_normal; 
uniform sampler2D map;
uniform vec3 light; // light direciton 

// Add output variable here
out vec4 out_color;

void main() {
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal); 
    vec3 L = normalize(light); 
    vec3 color = materialColor*clamp(dot(N, L), 0, 1);
    out_color = vec4(color, 1);
}
'''

bound = assimpObject.bound
width = 840
height = 480

displacement_vector = 2*bound.radius*glm.vec3(0,0,1) #glm.rotate(glm.vec3(0,1,0), glm.radians(60), glm.vec3(1,0,0)) #
    
target_point = glm.vec3(bound.center)
up_vector = glm.vec3(0,1,0)

### View volume parameters
fov_radian = glm.radians(45) # In radian
aspect = width/height
near = bound.radius
far = 3*bound.radius
perspectiveMatrix = glm.perspective(fov_radian, aspect, near, far)

pygame.init() # Initlizes its different modules. Display module is one of them.

pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE) 
pygame.display.set_mode((width, height), flags= pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
pygame.display.set_caption(title = "flying snoopy YAS")
gl = moderngl.get_context() # Get Previously created context.
gl.info["GL_VERSION"]

program = gl.program(
    vertex_shader= vertex_shader_code,
    fragment_shader= fragment_shader_code
)
format = "3f 3f 2f"
variables = ["position", "normal", "uv"]
renderables = assimpObject.getRenderables(gl, program, format, variables)

samplers = assimpObject.getSamplers(gl)

running = True
clock = pygame.time.Clock()
alpha = 0
pause = True
gl.enable(gl.DEPTH_TEST)
while running:   
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    # event.key == 27 means Escape key is pressed.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif (event.type ==  pygame.KEYDOWN):
            if  event.key == 27:
                running = False
            elif event.key == pygame.K_p:
                pause = not pause
        elif (event.type == pygame.WINDOWRESIZED):
            width = event.x
            height = event.y
            perspectiveMatrix = glm.perspective(fov_radian, width/height, near, far)

    # create the aspect ratio correction matrix
    new_displacement_vector = glm.rotate(displacement_vector, glm.radians(alpha), glm.vec3(0,1,0))
    eye_point = target_point + new_displacement_vector
    viewMatrix = glm.lookAt(eye_point, target_point, up_vector)

    
    ### Add render code below
    # Clear the display window with the specified R, G, B values using function ctx.clear(R, G, B)
    gl.clear(0.5,0.5,0.0)
    
    # Make one or more Render Calls to instruct the GPU to render by executing the shader program with the provided data.
    program["view"].write(viewMatrix)
    program["perspective"].write(perspectiveMatrix)
    program["light"].write(new_displacement_vector)
    assimpObject.render(program, renderables, samplers, M = glm.mat4(1))
    
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 10
    if not pause:
        alpha = alpha + 1
        if alpha > 360:
            alpha = 0
    
pygame.display.quit()