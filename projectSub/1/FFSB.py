import pygame
from pygame.locals import *
import moderngl
import numpy as np

# --- Initialize window + ModernGL ---
pygame.init()
screen = pygame.display.set_mode((900, 600), DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()
ctx.enable(moderngl.DEPTH_TEST)

# --- Load cube map images using only pygame ---
faces = [
    "Footballfield/posx.jpg",
    "Footballfield/negx.jpg",
    "Footballfield/posy.jpg",
    "Footballfield/negy.jpg",
    "Footballfield/posz.jpg",
    "Footballfield/negz.jpg",
]

# Load each face using pygame (no PIL or os)
surf = pygame.image.load(faces[0])
width, height = surf.get_size()
cubemap = ctx.texture_cube((width, height), 3, None)

for i, path in enumerate(faces):
    img = pygame.image.load(path).convert()
    raw = pygame.image.tostring(img, "RGB", False)  # upright
    cubemap.write(i, raw)

cubemap.build_mipmaps()
cubemap.use(0)

# --- Skybox cube geometry ---
vertices = np.array([
    -1, -1, -1,  1, -1, -1,  1,  1, -1,  -1,  1, -1,
    -1, -1,  1,  1, -1,  1,  1,  1,  1,  -1,  1,  1,
], dtype='f4')

indices = np.array([
    0,1,2, 2,3,0,   # back
    4,5,6, 6,7,4,   # front
    0,4,7, 7,3,0,   # left
    1,5,6, 6,2,1,   # right
    3,2,6, 6,7,3,   # top
    0,1,5, 5,4,0    # bottom
], dtype='i4')

vbo = ctx.buffer(vertices.tobytes())
ibo = ctx.buffer(indices.tobytes())

prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec3 in_vert;
        out vec3 texcoord;
        void main() {
            texcoord = in_vert;
            gl_Position = vec4(in_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        in vec3 texcoord;
        out vec4 fragColor;
        uniform samplerCube sky;
        uniform bool show_skybox;
        void main() {
            if (show_skybox)
                fragColor = texture(sky, texcoord);
            else
                fragColor = vec4(0.5, 0.5, 0.0, 0.0); // gray fallback
        }
    '''
)

vao = ctx.vertex_array(prog, [(vbo, "3f", "in_vert")], ibo)
prog["sky"].value = 0
show_skybox = True  # start with the football field visible

# --- Main loop ---
running = True
clock = pygame.time.Clock()

while running:
    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            running = False
        elif e.type == KEYDOWN and e.key == K_s:
            show_skybox = not show_skybox  # toggle background

    ctx.clear(0.5, 0.5, 0.0)
    prog["show_skybox"].value = show_skybox
    vao.render(moderngl.TRIANGLES)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
