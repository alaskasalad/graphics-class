import pygame
import math
import moderngl
import numpy as np

WIDTH = 600
HEIGHT = 600

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
pygame.display.set_caption(title="star practice")
gl = moderngl.get_context()

