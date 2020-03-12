#version 450

layout(location = 0) in vec3 position;  // vertex position
layout(location = 1) in vec3 normal; // normal vector
layout(location = 0) out vec3 v_normal;

// in the book we have
// - model-view-matrix
// - model-view-projection-matrix
// - normal-matrix

layout(set = 0, binding = 0) uniform Data {
    mat4 world;  // is this what is called the model-view projection matrix?  The model-view projection matrix transforms from world-space to view-space (clip-space?)
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * 3 * normal;
    // v_normal = inverse(mat3(worldview)) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 0.5);
}


// gl_Position is lke an out variable but it's built in and doesn't need to be declared as such. (?)  It's called a built-in variable.


// "Vertex shaders have the following predefined outputs"
