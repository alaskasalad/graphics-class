import pygame
import moderngl
import numpy as np 
import math

WIDTH = 600
HEIGHT = 600
size = (WIDTH, HEIGHT)
FPS = 30

with open ("vert.glsl") as vertFile:
    vertexShaderCode = vertFile.read()
with open ("frag.glsl") as fragFile:
    fragmentShaderCode = fragFile.read()

pygame.init()
 
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

pygame.display.set_mode((HEIGHT, WIDTH), flags= pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption(title="moving practice")
gl = moderngl.get_context()

position_data = [
    #x    y 
    0.0, 0.07,
    0.07, 0.0,
    -0.07, 0.0,

    0.0, -0.07,
    0.07, 0.0,
    -0.07, 0.0
]
vertexPosition = np.array(position_data).astype("f4")
positionBuffer = gl.buffer(vertexPosition)

program = gl.program (
    vertex_shader=vertexShaderCode,
    fragment_shader=fragmentShaderCode
)

renderable = gl.vertex_array(program,
    [( positionBuffer, "2f", "position")]
)

# grid on the screen 
step = 2.0 / FPS
lines = []

# vertical lines
x = -1.0
while x <= 1.0:
    lines.extend([x, -1.0, x, 1.0])
    x += step

# horizontal lines
y = -1.0
while y <= 1.0:
    lines.extend([-1.0, y, 1.0, y])
    y += step

lines = np.array(lines, dtype="f4")
vbo = gl.buffer(lines.tobytes())
vao = gl.simple_vertex_array(program, vbo, "position")

running = True 
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
       if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == 27 ):
           running = False
    gl.clear(1,1,1) # black background 

    program['color'].value = (0,0,0) # white lines 
    vao.render(mode=gl.LINES)

    program['color'].value = (1.0, 1.0, 0.0)  # yellow
    renderable.render()

    pygame.display.flip()
    clock.tick(10)

pygame.quit()