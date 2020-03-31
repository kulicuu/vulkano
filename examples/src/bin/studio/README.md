# Studio


## Description

This is an experiment studying with the runtime-shader example, in order to make a shader-studio mode where shaders can be reloaded dynamically on file change.  The techniques developed here will be integrated into Scratch or a succeeding project.


## Goals

1. Live shader reload on file watch with immediate screen render effect.  **Done**

2. Interface to adjust the application defined transformations (matrices generally, the ones that live in application/Rust logic rather than GPU/shader logic) live during runtime.  This is pretty much the same as input for a game, except we'll be using it in a studio way to enable the developer to tranfer through various relevant morphisms.


#### Note:

Unlike the runtime-shader example, this should be run from the vulkano root.  
