import contextlib

import numpy as np
from OpenGL.GL import *

class GL_BasicShader:
  def __init__(self):
    # Basic vertex shader
    vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec3 aColor;
        out vec3 vertexColor;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            vertexColor = aColor;
        }
        """

    # Basic fragment shader
    fragment_shader_source = """
        #version 330 core
        in vec3 vertexColor;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(vertexColor, 1.0);
        }
        """

    # Compile shaders and create program
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    self.shader_program = shader_program
    self.vertex_shader = vertex_shader
    self.fragment_shader = fragment_shader

  def __del__(self):
    glDeleteProgram(self.shader_program)
    glDeleteShader(self.vertex_shader)
    glDeleteShader(self.fragment_shader)

  @contextlib.contextmanager
  def bind(self):
    glUseProgram(self.shader_program)
    yield
    glUseProgram(0)


class GL_BasicDrawing:

  def __init__(self):
    self.shader = GL_BasicShader()

  def draw_points(self,
                  vertices: list[tuple[float]],
                  colors: list[tuple[float]],
                  point_size=5.0):
    with self.shader.bind(), bind_vertices_colors(vertices, colors):
      glPointSize(point_size)
      glDrawArrays(GL_POINTS, 0, len(vertices))

  def draw_lines(self,
                 vertices: list[tuple[float]],
                 colors: list[tuple[float]],
                 line_width=1.0):
    with self.shader.bind(), bind_vertices_colors(vertices, colors):
      gl_version = glGetString(GL_VERSION).decode("utf-8")
      major_version = int(gl_version.split(".")[0])
      minor_version = int(gl_version.split(".")[1].split(" ")[0])
      if major_version == 4 and minor_version < 2:
        # glLineWidth with width != 1 is not supported in OpenGL 4.1
        glLineWidth(1)
      else:
        glLineWidth(line_width)

      glDrawArrays(GL_LINES, 0, len(vertices))


@contextlib.contextmanager
def bind_vertices_colors(vertices: list[tuple[float]],
                          colors: list[tuple[float]]):
  vertex_data = np.array(vertices, dtype=np.float32)
  color_data = np.array(colors, dtype=np.float32)

  vao = glGenVertexArrays(1)
  glBindVertexArray(vao)

  vbo = glGenBuffers(1)
  glBindBuffer(GL_ARRAY_BUFFER, vbo)
  glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
  glEnableVertexAttribArray(0)
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

  cbo = glGenBuffers(1)
  glBindBuffer(GL_ARRAY_BUFFER, cbo)
  glBufferData(GL_ARRAY_BUFFER, color_data.nbytes, color_data, GL_STATIC_DRAW)
  glEnableVertexAttribArray(1)
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

  yield

  glDisableVertexAttribArray(0)
  glDisableVertexAttribArray(1)
  glBindBuffer(GL_ARRAY_BUFFER, 0)
  glBindVertexArray(0)

  glDeleteBuffers(1, [vbo])
  glDeleteBuffers(1, [cbo])
  glDeleteVertexArrays(1, [vao])

