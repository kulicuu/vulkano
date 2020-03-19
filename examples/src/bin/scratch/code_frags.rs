


















// if counter == 5 {
//     for jdx in 0..vertices_count {
//         let cursor = &mesh.positions[(jdx * 3)..((jdx * 3) + 3)];
//         let normal_cursor = &mesh.normals[(jdx * 3)..((jdx * 3) + 3)];
//         // let indices_cursor = &mesh.indices[(jdx * 3)..((jdx * 3) + 3)];
//         // println!("aeuauejjj :: 3939 {:?}", cursor[0]);
//         vertices.push(Vertex { position: (cursor[0], cursor[1], cursor[2]) });
//         normals.push(Normal { normal: (normal_cursor[0], normal_cursor[1], normal_cursor[2])});
//         // indices.push(indices_cursor[0]);
//         // indices.push(indices_cursor[1]);
//         // indices.push(indices_cursor[2]);
//
//     }
//
// }















for (i, m) in models.iter().enumerate() {
    let mesh = &m.mesh;
    println!("model[{}].name = \'{}\'", i, m.name);
    println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);


    // println!("{:?}", mesh);
    // println!("{:?}", mesh.positions);

    let INDICES = mesh.indices.clone();
    let VERTICES = mesh.vertices.






    println!("Size of model[{}].indices: {}", i, mesh.indices.len());
    for f in 0..mesh.indices.len() / 3 {
    	println!("    idx[{}] = {}, {}, {}.", f, mesh.indices[3 * f],
    		mesh.indices[3 * f + 1], mesh.indices[3 * f + 2]);
    }

    Normals and texture coordinates are also loaded, but not printed in this example
    println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
    assert!(mesh.positions.len() % 3 == 0);
    for v in 0..mesh.positions.len() / 3 {
    	println!("    v[{}] = ({}, {}, {})", v, mesh.positions[3 * v],
    		mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]);
    }




}
























// for (i, m) in materials.iter().enumerate() {
// 	println!("material[{}].name = \'{}\'", i, m.name);
// 	println!("    material.Ka = ({}, {}, {})", m.ambient[0], m.ambient[1],
// 		m.ambient[2]);
// 	println!("    material.Kd = ({}, {}, {})", m.diffuse[0], m.diffuse[1],
// 		m.diffuse[2]);
// 	println!("    material.Ks = ({}, {}, {})", m.specular[0], m.specular[1],
// 		m.specular[2]);
// 	println!("    material.Ns = {}", m.shininess);
// 	println!("    material.d = {}", m.dissolve);
// 	println!("    material.map_Ka = {}", m.ambient_texture);
// 	println!("    material.map_Kd = {}", m.diffuse_texture);
// 	println!("    material.map_Ks = {}", m.specular_texture);
// 	println!("    material.map_Ns = {}", m.normal_texture);
// 	println!("    material.map_d = {}", m.dissolve_texture);
// 	for (k, v) in &m.unknown_param {
// 		println!("    material.{} = {}", k, v);
// 	}
// }
