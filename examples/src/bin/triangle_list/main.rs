// If the data size for this gets too big, it might be necessary to have a thread doing this.




extern crate srtm;


use std::fs;
use std::fs::File;
use std::io::prelude::*;


use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad};




fn main() {
    let tile = srtm::Tile::from_file("./examples/src/bin/scratch/N57E016.hgt").unwrap();
    // let bounds = tile.extent() - 3;
    let bounds = 600;
    let mut vertex_str = String::from("Vertices/Positions:Start");
    let mut idx_str = String::from("\nIndices:Start\n");
    let mut normals_str = String::from("\nNormals:Start");
    let mut idx = 0;


    while idx < bounds {
        let mut jdx = 0;
        while jdx < bounds {

            let point = tile.get(idx, jdx);

            let vertex : Vector3<f64> = Vector3 { x: f64::from(idx), y: f64::from(jdx), z: f64::from(point) };
            let neighbor : Vector3<f64>;

            if jdx % 2 == 0 {
                let neighbor_point = tile.get(idx, jdx + bounds);
                neighbor = Vector3 {x: f64::from(idx), y: f64::from(jdx + bounds), z: f64::from(neighbor_point)};
            } else {
                let neighbor_point = tile.get(idx, jdx - 1);
                neighbor = Vector3 { x: f64::from(idx), y: f64::from(jdx + bounds), z: f64::from(neighbor_point) };
            }

            let pre_normal = vertex.cross(neighbor);
            let magnitude = ((pre_normal.x * pre_normal.x) + (pre_normal.y * pre_normal.y) + (pre_normal.z * pre_normal.z)).sqrt();
            let normal = Vector3 {x: pre_normal.x / magnitude, y: pre_normal.y / magnitude, z: pre_normal.z / magnitude };
            let mut normal_cursor = String::from("\n");
            normal_cursor.insert_str(normal_cursor.len(), &normal.x.to_string());
            normal_cursor.insert_str(normal_cursor.len(), " ");
            normal_cursor.insert_str(normal_cursor.len(), &normal.y.to_string());
            normal_cursor.insert_str(normal_cursor.len(), " ");
            normal_cursor.insert_str(normal_cursor.len(), &normal.z.to_string());
            normals_str.insert_str(normals_str.len(), &normal_cursor);


            let mut position = String::from("\n");
            position.insert_str(position.len(), &f64::from(idx).to_string());
            position.insert_str(position.len(), " ");
            position.insert_str(position.len(), &f64::from(jdx).to_string());
            position.insert_str(position.len(), " ");
            position.insert_str(position.len(), &f64::from(point).to_string());

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
                if ndx == (bounds - 1) {
                    idx_str.insert_str(idx_str.len(), "\n");
                    idx_str.insert_str(idx_str.len(), &(((bounds * kdx) + ndx) + bounds).to_string());
                }
                ndx += 1;
            }
            idx_str.insert_str(idx_str.len(), "\n");
            mdx += 1;
        }
        kdx += 1;
    }

    vertex_str.insert_str(vertex_str.len(), "\nVertices_End");
    let mut mesh_str = String::new();
    mesh_str.insert_str(0, &vertex_str);
    mesh_str.insert_str(mesh_str.len(), &idx_str);
    mesh_str.insert_str(mesh_str.len(), "IndicesEnd\n");
    mesh_str.insert_str(mesh_str.len(), &normals_str);
    mesh_str.insert_str(mesh_str.len(), "NormalsEnd\n");
    fs::write("./examples/src/bin/triangle_list/mesh.txt", mesh_str);
}
