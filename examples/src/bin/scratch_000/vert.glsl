uniform mat4 uModelViewMatrix;
uniform mat4 uModelViewProjectionMatrix;
uniform mat3 uNormalMatrix;

in vec4 aVertex;
in vec4 aTexCoord0;
in vec4 aColor;
in vec3 aNormal;

out vec4 vColor;
out float vLightIntensity;
out vec2 vST;

const vec3 LIGHTPOS = vec3( 0. , 0. , 10. );

void main() {
    vec3 transNorm = normalize( vec3 ( uNormalMatrix * aNormal ));
    vec3 ECposition = vec3( uModelViewMatrix * aVertex );
    vLightIntensity = dot( normalize(LIGHTPOS - ECPosition), transNorm );
    vLightIntensity = clamp( .3 + abs( vLightIntensity ), 0. , 1. );

    vST = aTexCoord0.st;
    vColor = aColor;
    gl_Position = uModelViewProjectionMatrix * aVertex;

}
