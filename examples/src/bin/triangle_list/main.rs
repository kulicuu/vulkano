// If the data size for this gets too big, it might be necessary to have a thread doing this.




extern crate srtm;


use std::fs;
use std::fs::File;
use std::io::prelude::*;





fn main() {

    let tile = srtm::Tile::from_file("./examples/src/bin/scratch/N57E016.hgt").unwrap();
    // let bounds = tile.extent();
    let bounds = 6;
    let mut vertex_str = String::from("Vertices/Positions:");
    let mut idx_str = String::from("Indices:\n");

    let mut idx = 0;



    while idx < bounds {
        let mut jdx = 0;
        while jdx < bounds {
            let point = tile.get(idx, jdx);
            let mut position = String::from("\n");
            position.insert_str(position.len(), &idx.to_string());
            position.insert_str(position.len(), " ");
            position.insert_str(position.len(), &jdx.to_string());
            position.insert_str(position.len(), " ");
            position.insert_str(position.len(), &point.to_string());

            vertex_str.insert_str(vertex_str.len(), &position);
            jdx = jdx + 1;
        }
        idx = idx + 1;
    }

    let mut kdx = 0;

    while kdx <= bounds {
        let mut mdx = 0;
        let mut ndx = 0;
        while ndx <= bounds - 1 {
            if mdx % 2 == 0 {
                idx_str.insert_str(idx_str.len(), &((bounds * kdx) + ndx).to_string());

            } else {
                idx_str.insert_str(idx_str.len(), &(((bounds * kdx) + ndx) + bounds).to_string());
                ndx += 1;
            }
            idx_str.insert_str(idx_str.len(), "\n");
            mdx += 1;
        }
        kdx += 1;
    }


    vertex_str.insert_str(vertex_str.len(), "\n\n\n");
    let mut mesh_str = String::new();
    mesh_str.insert_str(0, &vertex_str);
    mesh_str.insert_str(mesh_str.len(), &idx_str);
    fs::write("./examples/src/bin/triangle_list/mesh.txt", mesh_str);
}
