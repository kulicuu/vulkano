#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view;
    v_normal = transpose(inverse(mat3(worldview))) * normal;

    mat4 magnify = mat4(1, 0, 0, 0,     0, 1, 0, 0,     0, 0, 1, 0,      0, 0, 0, 1/30.0);
    mat4 rotate = mat4(1, 0, 0, 0,   0, 0, 1, 0,    0, -1, 0, 0,    0, 0, 0, 1);
    gl_Position = uniforms.proj * worldview * magnify * rotate * vec4(position, 1.0);
}
