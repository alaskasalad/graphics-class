import pygame
import moderngl
import numpy

WIDTH = 600
HEIGHT = 600

with open ("vert.glsl") as vertFile:
    vertexShaderCode = vertFile.read()
with open ("frag.glsl") as fragFile:
    fragmentShaderCode = fragFile.read()

pygame.init()

#its not showing with this ermmm .. 
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

pygame.display.set_mode((HEIGHT, WIDTH), flags= pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption(title="inverse triangle")
gl = moderngl.get_context()

position_data = [
    #x    y 
    0.0, -0.8,
    0.8, 0.8,
    -0.8, 0.8
]
vertextPositions = numpy.array(position_data).astype("float32")
positionBuffer = gl.buffer(vertextPositions)

program = gl.program (
    vertex_shader= vertexShaderCode,
    fragment_shader= fragmentShaderCode
)

renderable = gl.vertex_array(program,
    [( positionBuffer, "2f", "position")]
)

running = True 
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
       if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == 27 ):
           running = False
    gl.clear(1, 1, 1)
    renderable.render()
    pygame.display.flip()
    clock.tick(10)

pygame.quit()

           