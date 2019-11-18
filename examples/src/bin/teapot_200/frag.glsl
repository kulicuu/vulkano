#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(1, 1, 1);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = vec3(1, 1, 1);
    vec3 regular_color = vec3(1, 1, 1);

    f_color = vec4(mix(dark_color, regular_color, brightness), 40);
}
